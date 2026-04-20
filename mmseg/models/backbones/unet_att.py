# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import UpConvBlock, Upsample

class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1,c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out

@MODELS.register_module()
class UNet_ATT(BaseModule):
    """UNet backbone with ECA added to encoder stages."""

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        # Initialize ECA
        self.eca = nn.ModuleList([ECA(c1=base_channels * 2**i, c2=base_channels * 2**i) for i in range(num_stages)])

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2**i,
                        skip_channels=base_channels * 2**(i - 1),
                        out_channels=base_channels * 2**(i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None,
                        dcn=None,
                        plugins=None))

            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dcn=None,
                    plugins=None))

            # Add ECA after the encoder block
            enc_conv_block.append(self.eca[i])
            self.encoder.append(nn.Sequential(*enc_conv_block))
            in_channels = base_channels * 2**i

    def forward(self, x):
        self._check_input_divisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs
        
    def train(self, mode=True):
        """Convert the model into training mode while keeping normalization layers frozen."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0), \
            "Input dimensions must be divisible by the overall downsample rate."

"""lite_edge_fusion_patch.py

Lightweight frequency‑aware fusion wrapper for UNet_ATT_DS.

* `LiteEdgeFusion` – HR/LR edge‑aware fusion block
* `UpFuseBlock`    – decoder block drop‑in replacement for `UpConvBlock`
* `attach_lite_fusion(model)` – runtime patcher (no source edits)
* `UNet_ATT_DS_LiteEdge` – subclass of original UNet that auto‑patches itself
  so you can simply set `type='UNet_ATT_DS_LiteEdge'` in the config.
"""

# ----------------------------------------------------------------------
# 1. LiteEdgeFusion
# ----------------------------------------------------------------------
class LiteEdgeFusion(nn.Module):
    def __init__(self, hr_c: int, lr_c: int, out_c: int):
        super().__init__()
        # LR blur (fixed box)
        self.blur = nn.Conv2d(lr_c, lr_c, 3, 1, 1, groups=lr_c, bias=False)
        with torch.no_grad():
            self.blur.weight.fill_(1 / 9)
        self.blur.requires_grad_(False)
        # HR edge + SE
        self.sobel = nn.Conv2d(hr_c, hr_c, 3, 1, 1, groups=hr_c, bias=False)
        k = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        with torch.no_grad():
            for c in range(hr_c):
                self.sobel.weight[c, 0] = k
        self.sobel.requires_grad_(False)
        se_mid = max(hr_c // 16, 4)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(hr_c, se_mid, 1), nn.ReLU(True),
                                nn.Conv2d(se_mid, hr_c, 1), nn.Sigmoid())
        # depthwise‑separable mixer
        tot = hr_c + lr_c
        self.mix_dw = nn.Conv2d(tot, tot, 3, 1, 1, groups=tot, bias=False)
        self.mix_pw = nn.Conv2d(tot, out_c, 1, bias=False)
        self.act = nn.ReLU(True)

    def forward(self, hr, lr):
        lr = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        lr = self.blur(lr)
        edge = self.sobel(hr)
        hr = hr + edge * self.se(edge)
        x = torch.cat([hr, lr], 1)
        return self.act(self.mix_pw(self.mix_dw(x)))

# ----------------------------------------------------------------------
# 2. UpFuseBlock – decoder replacement
# ----------------------------------------------------------------------
class UpFuseBlock(nn.Module):
    """功能等价于 UpConvBlock，但用 LiteEdgeFusion 取代 concat-conv。"""
    def __init__(self, conv_block_cls,              # BasicConvBlock 类
                 in_channels, skip_channels, out_channels,
                 num_convs=2):
        super().__init__()
        # 1) 高-低分辨率特征融合
        self.fuse = LiteEdgeFusion(skip_channels, in_channels, out_channels)
        # 2) 融合后再走与原 UpConvBlock 相同的 conv_block
        self.post_conv = conv_block_cls(out_channels, out_channels,
                                        num_convs=num_convs)
    def forward(self, skip, x):
        x = self.fuse(skip, x)        # 空间对齐在 LiteEdgeFusion 内部完成
        return self.post_conv(x)


# ----------------------------------------------------------------------
# 3. Runtime patcher
# ----------------------------------------------------------------------

def _first_last_conv(mod: nn.Module):
    first = last = None
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            if first is None:
                first = m
            last = m
    if first is None or last is None:
        raise RuntimeError('No Conv2d found in module')
    return first, last

def attach_lite_fusion(model: nn.Module):
    """把 decoder 中 *所有* UpConvBlock → UpFuseBlock（不依赖私有属性）"""
    if not hasattr(model, 'decoder'):
        raise AttributeError('model has no decoder attr')

    for i, old in enumerate(model.decoder):
        # ① 推断 skip_channels & out_channels
        first_conv, last_conv = _first_last_conv(old.conv_block)
        out_channels  = last_conv.out_channels            # 与原 UpConvBlock 一致
        skip_channels = first_conv.in_channels // 2       # 因为 concat 了 2×skip

        # ② 推断 high-level in_channels (来自 upsample之前的特征)
        up_first_conv, _ = _first_last_conv(old.upsample)
        in_channels = up_first_conv.in_channels

        # ③ BasicConvBlock 类 & 卷积层数
        conv_block_cls = type(old.conv_block)
        num_convs = len(old.conv_block.convs) if hasattr(old.conv_block, 'convs') else 2

        # ④ 替换
        model.decoder[i] = UpFuseBlock(conv_block_cls,
                                       in_channels, skip_channels,
                                       out_channels, num_convs=num_convs)
    return model

@MODELS.register_module()
class UNet_ATT_LiteEdge(UNet_ATT):
    """UNet_ATT_DS with LiteEdgeFusion decoder (auto‑patched)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        attach_lite_fusion(self)