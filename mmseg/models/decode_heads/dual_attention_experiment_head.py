# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from torch import nn

from mmseg.registry import MODELS
from .da_head import CAM, PAM
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class DualAttentionExperimentHead(BaseDecodeHead):
    """Dual-attention experimental head for forest experiments.

    Supported modes:
    - 'cam_to_pam'
    - 'pam_to_cam'
    - 'parallel_concat'
    - 'parallel_residual'
    - 'gated_fusion'
    """

    def __init__(self, pam_channels=64, mode='parallel_concat', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.valid_modes = {
            'cam_to_pam', 'pam_to_cam', 'parallel_concat', 'parallel_residual',
            'gated_fusion'
        }
        if self.mode not in self.valid_modes:
            raise ValueError(f'Unsupported mode: {self.mode}')

        self.in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.pam = PAM(self.channels, pam_channels)
        self.cam = CAM()

        self.pam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.mode in {'parallel_concat', 'parallel_residual'}:
            self.fuse_conv = ConvModule(
                self.channels * 2,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.fuse_conv = None

        if self.mode == 'gated_fusion':
            self.gate_conv = nn.Conv2d(self.channels * 2, 2, kernel_size=1)
        else:
            self.gate_conv = None

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        feat = self.in_conv(x)

        if self.mode == 'cam_to_pam':
            out = self.cam(feat)
            out = self.cam_out_conv(out)
            out = self.pam(out)
            out = self.pam_out_conv(out)
            return self.cls_seg(out)

        if self.mode == 'pam_to_cam':
            out = self.pam(feat)
            out = self.pam_out_conv(out)
            out = self.cam(out)
            out = self.cam_out_conv(out)
            return self.cls_seg(out)

        pam_feat = self.pam_out_conv(self.pam(feat))
        cam_feat = self.cam_out_conv(self.cam(feat))

        if self.mode == 'parallel_concat':
            out = self.fuse_conv(torch.cat([pam_feat, cam_feat], dim=1))
            return self.cls_seg(out)

        if self.mode == 'parallel_residual':
            out = self.fuse_conv(torch.cat([pam_feat, cam_feat], dim=1))
            out = out + feat
            return self.cls_seg(out)

        # gated_fusion
        cat_feat = torch.cat([pam_feat, cam_feat], dim=1)
        gates = torch.softmax(self.gate_conv(cat_feat), dim=1)
        out = gates[:, 0:1] * pam_feat + gates[:, 1:2] * cam_feat
        return self.cls_seg(out)
