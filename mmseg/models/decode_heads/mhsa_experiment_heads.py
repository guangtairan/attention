# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import nn

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class MHSA2DExperimentHead(BaseDecodeHead):
    """Stage4 MHSA head with conv-style QKV projections.

    This variant keeps a 2D feature-map style implementation and applies
    attention over spatial locations internally.
    """

    def __init__(self,
                 num_heads=8,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        if self.channels % num_heads != 0:
            raise ValueError('channels must be divisible by num_heads')

        self.num_heads = num_heads
        self.head_dim = self.channels // num_heads
        self.scale = self.head_dim**-0.5

        self.in_conv = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.q_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.k_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.v_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.out_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_conv = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _to_multihead(self, x):
        b, c, h, w = x.shape
        n = h * w
        x = x.reshape(b, self.num_heads, self.head_dim, n)
        return x, h, w, n

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        feat = self.in_conv(x)
        b = feat.shape[0]

        q, h, w, n = self._to_multihead(self.q_proj(feat))
        k, _, _, _ = self._to_multihead(self.k_proj(feat))
        v, _, _, _ = self._to_multihead(self.v_proj(feat))

        # q: B, heads, N, dim; k: B, heads, dim, N
        q = q.permute(0, 1, 3, 2)
        attn = torch.matmul(q, k) * self.scale
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        # v: B, heads, dim, N -> B, heads, N, dim
        v = v.permute(0, 1, 3, 2)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.channels, h, w)
        out = self.proj_drop(self.out_proj(out))
        out = out + feat
        out = self.out_conv(out)
        return self.cls_seg(out)


@MODELS.register_module()
class MHSATokenExperimentHead(BaseDecodeHead):
    """Stage4 MHSA head with explicit flatten-token -> MHSA -> reshape."""

    def __init__(self,
                 num_heads=8,
                 mlp_ratio=4.0,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        if self.channels % num_heads != 0:
            raise ValueError('channels must be divisible by num_heads')

        self.in_conv = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.norm1 = nn.LayerNorm(self.channels)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True)
        self.drop1 = nn.Dropout(proj_drop)

        hidden = int(self.channels * mlp_ratio)
        self.norm2 = nn.LayerNorm(self.channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, self.channels),
            nn.Dropout(proj_drop))

        self.out_conv = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        feat = self.in_conv(x)
        b, c, h, w = feat.shape
        n = h * w

        tokens = feat.flatten(2).transpose(1, 2)  # B, N, C

        t = self.norm1(tokens)
        attn_out, _ = self.mhsa(t, t, t, need_weights=False)
        tokens = tokens + self.drop1(attn_out)

        tokens = tokens + self.ffn(self.norm2(tokens))
        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        out = self.out_conv(out)
        return self.cls_seg(out)
