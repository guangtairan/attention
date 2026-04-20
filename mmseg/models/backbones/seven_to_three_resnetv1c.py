# mmseg/models/backbones/seven_to_three_resnetv1c.py
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class SevenToThreeResNetV1c(BaseModule):
    """Adapter: 7ch input -> 3ch, then feed into a normal 3ch ResNetV1c.

    This keeps ImageNet pretrained weights fully usable (3ch conv1).
    """

    def __init__(self,
                #  in_channels: int = 7,
                 in_channels: int = 5,
                 out_channels: int = 3,
                 adapter_bias: bool = False,
                 adapter_init: str = 'first3_identity',  # or 'kaiming'
                 backbone_cfg: dict = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert backbone_cfg is not None, "Please provide backbone_cfg for the inner ResNetV1c."
        assert out_channels == 3, "This wrapper is designed to output 3 channels for ImageNet pretrained ResNet."

        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=adapter_bias)

        # Build the inner 3-channel backbone (ResNetV1c)
        # Make sure inner backbone takes 3-channel input
        backbone_cfg = backbone_cfg.copy()
        backbone_cfg.setdefault('in_channels', 3)
        self.backbone = MODELS.build(backbone_cfg)

        self._init_adapter(adapter_init)

    def _init_adapter(self, mode: str):
        # Make the adapter stable at start:
        # - first3_identity: copy first 3 bands to RGB-like channels, other bands start at 0
        # - kaiming: normal conv init
        if mode == 'first3_identity':
            nn.init.zeros_(self.adapter.weight)
            with torch.no_grad():
                # Map band0->R, band1->G, band2->B (you can change order if needed)
                for c in range(min(3, self.adapter.in_channels)):
                    self.adapter.weight[c, c, 0, 0] = 1.0
            if self.adapter.bias is not None:
                nn.init.zeros_(self.adapter.bias)
        elif mode == 'kaiming':
            nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
            if self.adapter.bias is not None:
                nn.init.zeros_(self.adapter.bias)
        else:
            raise ValueError(f"Unknown adapter_init: {mode}")

    def forward(self, x):
        # x: (N,7,H,W)
        x = self.adapter(x)
        return self.backbone(x)
