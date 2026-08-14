"""MixSTE-based Siamese encoder for the motion-alignment ablation.

Faithful re-implementation of MixSTE (Zhang et al., "MixSTE: Seq2seq
Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video",
CVPR 2022). The transformer stack mirrors
``other_methods/MixSTE/common/model_cross.py :: MixSTE2`` exactly,
including the alternating ``STE`` (spatial) and ``TTE`` (temporal)
blocks across ``depth`` iterations:

* ``STE_forward``: per-frame transformer over joint tokens.
* ``TTE_forward``: per-joint transformer over frame tokens.
* ``ST_forward``: ``depth - 1`` further alternating STE/TTE iterations.

Two adaptations vs. the original:

1. Input adapter — original is ``(B, F, J, 2)`` 2D pose; ours is
   ``(N, 1, S=75, F=117)`` 3D mocap. We reshape to
   ``(N, 75, 39, 3)`` (``in_chans=3, num_joints=39``) so the
   ``Spatial_patch_to_embedding`` linear sees one marker (3 channels)
   per token.
2. Output adapter — MixSTE outputs ``(B, F, J, 3)`` per-frame 3D pose.
   For Siamese alignment we need a fixed-size embedding, so we replace
   the per-joint regression head with mean-pooling over (frames, joints)
   followed by ``LayerNorm + Linear(embed_dim_ratio -> out_dim)``.

The ``Mlp``/``Attention``/``Block`` blocks are the same minimal versions
already used in ``model_poseformer.py``; ``DropPath`` is reimplemented
locally so we don't depend on ``timm`` and we use native ``torch``
reshapes instead of einops to keep the dependency footprint identical
to the rest of ``motion_alignment/``.

Default hyper-parameters match the MixSTE paper / its
``common/arguments.py`` (``-cs 512 -dep 8``) and ``run.py`` line 230-234
(``num_heads=8, mlp_ratio=2., qkv_bias=True, drop_path_rate=0.1``).
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from model_poseformer import Attention, DropPath, Mlp  # noqa: F401  (reuse)


class _Block(nn.Module):
    """Plain transformer block (matches MixSTE's ``Block`` w/o changedim)."""

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
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixSTEEncoder(nn.Module):
    """MixSTE encoder that produces a fixed-size embedding for Siamese alignment.

    Input: ``(N, 1, num_frame, num_joints*in_chans)``. Output: ``(N, out_dim)``.

    Parameters
    ----------
    num_frame, num_joints, in_chans
        Pose-input dimensions. Defaults match the mocap pipeline
        (75 frames, 39 markers, xyz).
    embed_dim_ratio
        Channel size (MixSTE calls it ``cs``). Default 512 from the
        published config.
    depth
        Alternating-STE/TTE iteration count. Default 8 (paper).
    num_heads, mlp_ratio, qkv_bias, drop_path_rate
        Same defaults as MixSTE's ``run.py`` ``model_pos_train``.
    out_dim
        Final embedding dim — set to 256 to match ``model.Encoder``.
    """

    def __init__(
        self,
        num_frame: int = 75,
        num_joints: int = 39,
        in_chans: int = 3,
        embed_dim_ratio: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_dim: int = 256,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # In MixSTE, embed_dim == embed_dim_ratio (no joint multiplication).
        embed_dim = embed_dim_ratio

        self.num_frame = num_frame
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.embed_dim_ratio = embed_dim_ratio
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.block_depth = depth

        # Spatial patch embedding (per-marker).
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Spatial (per-frame, joints as tokens).
        self.STEblocks = nn.ModuleList([
            _Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        # Temporal (per-joint, frames as tokens).
        self.TTEblocks = nn.ModuleList([
            _Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # Embedding head: pool over (frames, joints) -> Linear to out_dim.
        # MixSTE's original head is Linear(embed_dim -> 3); we replace it
        # with an embedding projection of the same input dim.
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        nn.init.trunc_normal_(self.Spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.Temporal_pos_embed, std=0.02)
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
    # Mirror MixSTE2.STE_forward / TTE_foward / ST_foward exactly,
    # with native torch reshapes instead of einops.
    # ------------------------------------------------------------------
    def _ste_forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(b, f, n, c) -> (b*n, f, cw)`` after first STE block."""
        b, f, n, c = x.shape
        x = x.reshape(b * f, n, c)
        x = self.Spatial_patch_to_embedding(x)
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)
        # (b*f, n, cw) -> (b, f, n, cw) -> (b, n, f, cw) -> (b*n, f, cw)
        x = x.reshape(b, f, n, -1).permute(0, 2, 1, 3).reshape(b * n, f, -1)
        return x

    def _tte_forward(self, x: torch.Tensor) -> torch.Tensor:
        """First temporal block. Input ``(b*n, f, cw)``."""
        x = x + self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)
        return x

    def _st_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remaining ``depth - 1`` alternating STE/TTE iterations.

        Input: ``(b, f, n, cw)``. Output: same shape.
        """
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            # STE pass.
            x = x.reshape(b * f, n, cw)
            x = self.STEblocks[i](x)
            x = self.Spatial_norm(x)
            # (b*f, n, cw) -> (b, n, f, cw) -> (b*n, f, cw)
            x = x.reshape(b, f, n, cw).permute(0, 2, 1, 3).reshape(b * n, f, cw)
            # TTE pass.
            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
            # (b*n, f, cw) -> (b, n, f, cw) -> (b, f, n, cw)
            x = x.reshape(b, n, f, cw).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 1, S, J*C) -> (N, out_dim)``."""
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # (N, S, J*C)
        b, f, fc = x.shape
        n = self.num_joints
        c = self.in_chans
        if fc != n * c:
            raise ValueError(
                f"Expected last dim {n * c} (={n} joints * {c} chans); got {fc}"
            )
        x = x.reshape(b, f, n, c)

        x = self._ste_forward(x)         # (b*n, f, cw)
        x = self._tte_forward(x)         # (b*n, f, cw)
        x = x.reshape(b, n, f, -1).permute(0, 2, 1, 3).contiguous()  # (b, f, n, cw)
        x = self._st_forward(x)          # (b, f, n, cw)

        # Pool over (frames, joints) -> single representation per sequence.
        x = x.mean(dim=(1, 2))           # (b, cw)
        x = self.head(x)                 # (b, out_dim)
        return x
