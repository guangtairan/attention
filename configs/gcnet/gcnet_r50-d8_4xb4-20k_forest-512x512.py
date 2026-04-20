_base_ = [
    '../_base_/models/gcnet_r50-d8.py',
    '../_base_/datasets/forest.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
custom_imports = dict(
    imports=['mmseg.datasets.transforms.debug_transforms'],
    allow_failed_imports=False
)
custom_imports = dict(
    imports=['mmseg.models.backbones.seven_to_three_resnetv1c'],
    allow_failed_imports=False
)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))


load_from =None
# load_from =r'E:\Desktop\mmsegmentation-1.2.2\danet_r50-d8_512x512_80k_ade20k_20200615_015125-edb18e08.pth'


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

