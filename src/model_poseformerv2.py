"""PoseFormerV2-based Siamese encoder for the motion-alignment ablation.

Faithful re-implementation of PoseFormerV2 (Zhao et al., "PoseFormerV2:
Exploring Frequency Domain for Efficient and Robust 3D Human Pose
Estimation", CVPR 2023). Mirrors
``other_methods/PoseFormerV2/common/model_poseformer.py ::
PoseTransformerV2`` exactly, including:

* Time-domain ``Joint_embedding`` over ``num_frame_kept`` center frames.
* Frequency-domain ``Freq_embedding`` over ``num_coeff_kept`` DCT
  coefficients (using ``torch_dct``).
* ``Spatial_blocks`` — per-frame spatial transformer over joint tokens.
* ``MixedBlock`` stack — joint attention over concatenated time+freq
  tokens with a split MLP (``mlp1`` on time tokens, ``FreqMlp`` =
  DCT-MLP-IDCT on freq tokens).
* Two ``Conv1d`` "weighted-mean" pools collapsing time and freq tokens,
  concatenated into the final embedding.

Two adaptations:

1. Input adapter — original is ``(B, F, J, 2)`` 2D pose; ours is
   ``(N, 1, S=75, F=117)`` (3D mocap). Reshape to
   ``(N, 75, 39, 3)`` (``in_chans=3, num_joints=39``).
2. Output adapter — original head is ``LayerNorm + Linear(2*embed_dim ->
   num_joints*3)`` for 3D-pose regression. We replace with
   ``LayerNorm + Linear(2*embed_dim -> 256)`` for the Siamese embedding.

Defaults match ``demo/vis.py``: ``embed_dim_ratio=32, depth=4,
number_of_kept_frames=27, number_of_kept_coeffs=27``. Original PoseFormerV2
took 243-frame inputs; we feed 75. ``kept_frames=27`` still picks the 27
center frames as in the published model.

``DropPath`` / ``Mlp`` / ``Attention`` / ``Block`` are imported from
``model_poseformer`` (they're identical between PoseFormer and
PoseFormerV2). ``FreqMlp`` and ``MixedBlock`` are added here.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch_dct as dct

from model_poseformer import Attention, Block, DropPath, Mlp


class FreqMlp(nn.Module):
    """DCT -> MLP -> IDCT (matches PoseFormerV2's ``FreqMlp`` verbatim)."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, f, c). DCT along the time axis.
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x


class MixedBlock(nn.Module):
    """PoseFormerV2's ``MixedBlock``: shared attention, split MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
        )
        self.norm3 = norm_layer(dim)
        self.mlp2 = FreqMlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # First half = freq tokens, second half = time tokens (matches
        # PoseFormerV2 exactly: the freq sequence is concatenated FIRST
        # in ``forward_features``).
        x1 = x[:, :f // 2] + self.drop_path(self.mlp1(self.norm2(x[:, :f // 2])))
        x2 = x[:, f // 2:] + self.drop_path(self.mlp2(self.norm3(x[:, f // 2:])))
        return torch.cat((x1, x2), dim=1)


class PoseFormerV2Encoder(nn.Module):
    """PoseFormerV2 encoder that produces a fixed-size embedding for Siamese alignment.

    Input: ``(N, 1, num_frame, num_joints*in_chans)``. Output:
    ``(N, out_dim)``.
    """

    def __init__(
        self,
        num_frame: int = 75,
        num_joints: int = 39,
        in_chans: int = 3,
        embed_dim_ratio: int = 32,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        number_of_kept_frames: int = 27,
        number_of_kept_coeffs: int = 27,
        out_dim: int = 256,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints

        self.num_frame = num_frame
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.embed_dim_ratio = embed_dim_ratio
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_frame_kept = number_of_kept_frames
        self.num_coeff_kept = number_of_kept_coeffs or number_of_kept_frames

        # Embeddings.
        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans * num_joints, embed_dim)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.blocks = nn.ModuleList([
            MixedBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # Two weighted-mean Conv1d pools (matches the original).
        self.weighted_mean = nn.Conv1d(
            in_channels=self.num_coeff_kept, out_channels=1, kernel_size=1,
        )
        self.weighted_mean_ = nn.Conv1d(
            in_channels=self.num_frame_kept, out_channels=1, kernel_size=1,
        )

        # Embedding head: 2*embed_dim -> out_dim (the original head was
        # 2*embed_dim -> num_joints*3).
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, out_dim),
        )

        nn.init.trunc_normal_(self.Spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.Temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.Temporal_pos_embed_, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Mirror PoseTransformerV2.Spatial_forward_features / forward_features
    # / forward, with native torch reshapes instead of einops.
    # ------------------------------------------------------------------
    def _spatial_forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """``(b, f, p, c) -> (b, F_kept, p*emb_ratio)`` over center frames."""
        b, f, p, _ = x.shape
        kept = self.num_frame_kept
        # Center indices (matches the original: arange((f-1)//2-kept//2,
        # (f-1)//2+kept//2+1)).
        start = (f - 1) // 2 - kept // 2
        index = torch.arange(start, start + kept, device=x.device)
        x = x[:, index]                                    # (b, kept, p, c)
        x = self.Joint_embedding(x.reshape(b * kept, p, -1))
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        # (b*kept, p, e) -> (b, kept, p*e)
        x = x.reshape(b, kept, -1)
        return x

    def _forward_features(self, x: torch.Tensor, spatial_feature: torch.Tensor) -> torch.Tensor:
        """Time-domain spatial features + DCT-domain features through MixedBlocks."""
        b, f, p, _ = x.shape
        kept = self.num_coeff_kept

        # DCT along the time axis. Original code:
        #   x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        # That's (b, p, c, f) -> DCT over last dim -> keep first `kept` coeffs.
        x = dct.dct(x.permute(0, 2, 3, 1))[..., :kept]      # (b, p, c, kept)
        x = x.permute(0, 3, 1, 2).contiguous().reshape(b, kept, -1)  # (b, kept, p*c)
        x = self.Freq_embedding(x)                           # (b, kept, embed_dim)

        spatial_feature = spatial_feature + self.Temporal_pos_embed
        x = x + self.Temporal_pos_embed_
        x = torch.cat((x, spatial_feature), dim=1)           # (b, kept+kept, embed_dim)

        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 1, S, J*C) -> (N, out_dim)``."""
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        b, f, fc = x.shape
        n = self.num_joints
        c = self.in_chans
        if fc != n * c:
            raise ValueError(
                f"Expected last dim {n * c} (={n} joints * {c} chans); got {fc}"
            )
        x = x.reshape(b, f, n, c)
        x_ = x.clone()

        spatial_feature = self._spatial_forward_features(x)        # (b, kept, embed_dim)
        x = self._forward_features(x_, spatial_feature)            # (b, 2*kept, embed_dim)

        # Two weighted-mean pools, concatenated -> (b, 1, 2*embed_dim).
        first = self.weighted_mean(x[:, : self.num_coeff_kept])    # (b, 1, embed_dim)
        second = self.weighted_mean_(x[:, self.num_coeff_kept:])   # (b, 1, embed_dim)
        x = torch.cat((first, second), dim=-1)                     # (b, 1, 2*embed_dim)

        x = self.head(x).view(b, -1)                               # (b, out_dim)
        return x
