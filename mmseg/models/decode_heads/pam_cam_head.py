# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .da_head import CAM, PAM
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PAMHead(BaseDecodeHead):
    """PAM-only decode head."""

    def __init__(self, pam_channels, **kwargs):
        super().__init__(**kwargs)
        self.pam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam = PAM(self.channels, pam_channels)
        self.pam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        feat = self.pam_in_conv(x)
        feat = self.pam(feat)
        feat = self.pam_out_conv(feat)
        return self.cls_seg(feat)


@MODELS.register_module()
class CAMHead(BaseDecodeHead):
    """CAM-only decode head."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam = CAM()
        self.cam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        feat = self.cam_in_conv(x)
        feat = self.cam(feat)
        feat = self.cam_out_conv(feat)
        return self.cls_seg(feat)
