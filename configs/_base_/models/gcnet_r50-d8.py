# # model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 144.75, 44.48, 95.20, 3108.432],
    std=[58.395, 57.12, 61.33, 49.11, 48.17, 57.375, 79.733],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255)
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=data_preprocessor,
#     pretrained='open-mmlab://resnet50_v1c',
#     backbone=dict(
#         type='ResNetV1c',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         dilations=(1, 1, 2, 4),
#         strides=(1, 2, 1, 1),
#         norm_cfg=norm_cfg,
#         norm_eval=False,
#         style='pytorch',
#         contract_dilation=True),
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    # ❗把这行去掉（外层不再用 pretrained 字段）
    # pretrained='open-mmlab://resnet50_v1c',

    backbone=dict(
        type='SevenToThreeResNetV1c',
        in_channels=7,
        adapter_init='first3_identity',

        backbone_cfg=dict(
            type='ResNetV1c',
            depth=50,
            in_channels=3,  # 这里固定 3
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,

            # ✅ 内层 ResNet 用 ImageNet 预训练（写 init_cfg 更稳）
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')
        )
    ),
    decode_head=dict(
        type='GCHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        ratio=1 / 4.,
        pooling_type='att',
        fusion_types=('channel_add', ),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
