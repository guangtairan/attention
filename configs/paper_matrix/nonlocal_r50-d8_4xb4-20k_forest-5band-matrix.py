_base_ = [
    '../_base_/models/nonlocal_r50-d8.py',
    '../_base_/datasets/forest.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

custom_imports = dict(
    imports=['mmseg.models.backbones.seven_to_three_resnetv1c'],
    allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[119.11, 142.18, 62.78, 22.214, 3109.56],
    std=[50.11, 49.54, 41.51, 5.5079, 83.73],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

loss_decode = [
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
]

aux_loss_decode = [
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    dict(type='DiceLoss', loss_name='loss_aux_dice', loss_weight=0.4)
]

model = dict(
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='SevenToThreeResNetV1c',
        in_channels=5,
        adapter_init='first3_identity',
        backbone_cfg=dict(
            type='ResNetV1c',
            depth=50,
            in_channels=3,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))),
    decode_head=dict(num_classes=2, norm_cfg=norm_cfg, loss_decode=loss_decode),
    auxiliary_head=dict(
        num_classes=2, norm_cfg=norm_cfg, loss_decode=aux_loss_decode))

