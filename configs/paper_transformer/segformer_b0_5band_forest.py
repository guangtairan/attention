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

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[119.11, 142.18, 62.78, 22.214, 3109.56],
    std=[50.11, 49.54, 41.51, 5.5079, 83.73],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

loss_decode = [
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
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
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=loss_decode),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=None,
    accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.adapter': dict(lr_mult=5.0),
            'backbone.backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'backbone.backbone.pos_block': dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.backbone.norm': dict(lr_mult=0.1, decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0))

train_dataloader = dict(batch_size=2)
work_dir = './work_dirs/segformer_b0_5band_forest'
