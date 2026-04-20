# model settings
# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='GN',num_groups=32,requires_grad=True)
# SegDataPreProcessor 在 train_step、val_step 和 test_step 中
# 将数据传送到指定设备，之后处理后的数据将被进一步传递给模型。


data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[119.11, 142.18, 62.78, 22.214, 3109.56],
    std=[50.11, 49.54, 41.51, 5.5079, 83.73],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    # ❗把这行去掉（外层不再用 pretrained 字段）
    # pretrained='open-mmlab://resnet50_v1c',

    backbone=dict(
        type='SevenToThreeResNetV1c',
        in_channels=5,
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
# model = dict(
#     type='EncoderDecoder',#segmentor决定的前向传播顺序
#     data_preprocessor=data_preprocessor,
#     pretrained='open-mmlab://resnet50_v1c',
#     # pretrained='open-mmlab://resnet50_v1c',
#     # pretrained=None,
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
    decode_head=dict(
        type='DAHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pam_channels=64,
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
