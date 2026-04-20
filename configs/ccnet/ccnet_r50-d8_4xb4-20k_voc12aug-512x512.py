_base_ = [
    '../_base_/models/ccnet_r50-d8.py',
    '../_base_/datasets/forest.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
load_from=r'E:\Desktop\mmsegmentation-1.2.2\ccnet_r50-d8_512x512_80k_ade20k_20200615_014848-aa37f61e.pth'
