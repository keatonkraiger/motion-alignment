"""PoseFormer-based Siamese encoder for the motion-alignment ablation.

Faithful re-implementation of the spatial+temporal transformer stack from
``other_methods/PoseFormer/common/model_poseformer.py`` (Zheng et al.,
"3D Human Pose Estimation with Spatial and Temporal Transformers", ICCV
2021), with two adaptations needed to drop into the existing Siamese
pipeline:

1. The original input is ``(B, num_frame, num_joints, 2)`` (2D joint
   coords). Our patches are ``(N, 1, S=75, F=117)`` with ``F = J*3 = 39*3``
   (3D mocap markers, xyz interleaved per the MATLAB
   ``extractMocapPatches_tmm.m``). We reshape to
   ``(N, 75, 39, 3)`` before the transformer so PoseFormer's spatial
   patch-to-embedding sees one marker (3 channels) per token. Frames are
   75 = 3 s @ 25 fps, exactly as in the MATLAB pipeline.

2. PoseFormer's final ``head`` projects to ``num_joints * 3`` for 3D pose
   regression. We discard it and append a small projection head
   ``LayerNorm + Linear(embed_dim -> 256)`` so the encoder output dim
   matches the existing CNN encoder (``model.Encoder``) and plugs
   straight into ``CrossCorrLoss`` and the cosine-distance DTW eval
   without changing the loss / eval scale.

The transformer ``Block`` / ``Attention`` / ``Mlp`` blocks are copied
verbatim from PoseFormer (only ``DropPath`` is reimplemented locally so we
don't have to depend on ``timm``). einops is also dropped in favour of
native ``torch`` reshapes to keep the dependency footprint identical to
the existing pipeline.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None):
        super().__init__()
        self.drop_prob = drop_prob or 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
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

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PoseFormerEncoder(nn.Module):
    """Spatio-temporal transformer encoder for Siamese motion alignment.

    Drop-in replacement for ``model.Encoder`` (same input/output API):
    input ``(N, 1, num_frame, num_joints * in_chans)`` -> output
    ``(N, out_dim)`` with ``out_dim`` defaulting to 256.
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

        # --- Spatial branch (per-frame, joints as tokens). -------------
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        # --- Temporal branch (per-sequence, frames as tokens). ---------
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
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
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # PoseFormer's "weighted mean" trick: a 1x1 Conv1d over the time axis
        # collapses ``num_frame`` tokens to a single representation token.
        self.weighted_mean = nn.Conv1d(
            in_channels=num_frame, out_channels=1, kernel_size=1,
        )

        # Embedding head (replaces PoseFormer's 3D-pose regression head).
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        # Init: faithful to PoseFormer (trunc-normal on Linears + pos embeds).
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
    def _spatial_forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, F, J) -> (B, F, J*embed_dim_ratio)`` (per-frame tokens)."""
        b, c, f, p = x.shape
        # (b, c, f, p) -> (b, f, p, c) -> (b*f, p, c)
        x = x.permute(0, 2, 3, 1).reshape(b * f, p, c)
        x = self.Spatial_patch_to_embedding(x)
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        # (b*f, p, e) -> (b, f, p, e) -> (b, f, p*e)
        x = x.reshape(b, f, x.shape[-2], x.shape[-1])
        x = x.reshape(b, f, -1)
        return x

    def _temporal_forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, F, embed_dim) -> (B, embed_dim)`` (sequence-level token)."""
        b = x.shape[0]
        x = x + self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        x = self.weighted_mean(x)  # (b, 1, embed_dim)
        return x.view(b, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode patches into a fixed-size representation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, 1, num_frame, num_joints * in_chans)``. Channels
            for each joint are interleaved: ``[m1x, m1y, m1z, m2x, ...]``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, out_dim)`` representation, ready for
            ``CrossCorrLoss`` / cosine DTW.
        """
        N, _, S, F = x.shape
        if S != self.num_frame:
            raise ValueError(
                f"Expected num_frame={self.num_frame} along dim 2, got {S}"
            )
        if F != self.num_joints * self.in_chans:
            raise ValueError(
                f"Expected last dim = {self.num_joints} joints x {self.in_chans} chans"
                f" = {self.num_joints * self.in_chans}, got {F}"
            )
        # (N, 1, S, F) -> (N, S, J, C) -> (N, C, S, J) (PoseFormer's input layout).
        x = x.view(N, S, self.num_joints, self.in_chans)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self._spatial_forward(x)
        x = self._temporal_forward(x)
        x = self.head(x)
        return x
