_base_ = [
    '../_base_/datasets/forest.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

custom_imports = dict(
    imports=['mmseg.models.backbones.seven_to_three_resnetv1c'],
    allow_failed_imports=False)

crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[119.11, 142.18, 62.78, 22.214, 3109.56],
    std=[50.11, 49.54, 41.51, 5.5079, 83.73],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa

loss_decode = [
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
]
aux_loss_decode = [
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    dict(type='DiceLoss', loss_name='loss_aux_dice', loss_weight=0.4)
]

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SevenToThreeResNetV1c',
        in_channels=5,
        adapter_init='first3_identity',
        backbone_cfg=dict(
            type='SwinTransformer',
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=96,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=backbone_norm_cfg,
            init_cfg=dict(type='Pretrained', checkpoint=pretrained))),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.5,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=loss_decode),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.5,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=aux_loss_decode),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Minimal transformer-specific optimizer override on top of schedule_20k.py
# Keep base lr from the unified schedule, only adjust parameter groups.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone.adapter': dict(lr_mult=5.0),
            'backbone.backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'backbone.backbone.absolute_pos_embed': dict(
                lr_mult=0.1, decay_mult=0.0),
            'backbone.backbone.relative_position_bias_table': dict(
                lr_mult=0.1, decay_mult=0.0),
            'backbone.backbone.norm': dict(lr_mult=0.1, decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0))
