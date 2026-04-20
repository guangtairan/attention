_base_ = [
    'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_'
    'ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        pretrain_img_size=224,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=2),
    auxiliary_head=dict(in_channels=768, num_classes=2))
