_base_ = ['./_common_5band.py']

loss_decode = [
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
]

model = dict(
    decode_head=dict(
        _delete_=True,
        type='DualAttentionExperimentHead',
        mode='parallel_residual',
        in_channels=2048,
        in_index=3,
        channels=512,
        pam_channels=64,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=loss_decode))

