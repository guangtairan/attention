_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/crack.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)

model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2),
    data_preprocessor=data_preprocessor,
)
 
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)