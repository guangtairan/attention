# dataset settings
custom_imports = dict(
    imports=['mmseg.datasets.transforms.debug_transforms'],
    allow_failed_imports=False
)

dataset_type = 'FORESTDataset'
data_root = 'data/dataset5pix/forest'
img_scale = (512 , 512)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    # dict(type='ScaleBand7To255'),
    # dict(type='DebugImgOnce'), 
    # dict(type='LoadImageFromFile')
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),## 变化图像大小，以进行数据增广
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),## 随机裁剪，以进行数据增广
    # dict(type='DebugGTOnce'),     检查0 1比率
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),## 随机光学变换，亮度、对比度等
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    # dict(type='ScaleBand7To255'),
    # dict(type='DebugImgOnce'), 
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # dict(type='DebugGTOnce'), 检查0 1比率
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
#对同一张图做几种增强（比如不同尺度 resize、水平翻转），
# 分别推理得到多个预测结果，然后把它们融合（平均/投票）得到更稳的最终结果
tta_pipeline = [
    dict(type='LoadTiffImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    
    batch_size=2,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    #DefaultSampler屏蔽了单进程训练与多进程训练的细节差异，使得单卡与多卡训练可以无缝切换
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='images/training',
                seg_map_path='annotations/training'),
            pipeline=train_pipeline))
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    #InfiniteSampler 会无限循环地从数据集中采样，适用于按iter迭代
    # 也就是说，它不会在一个epoch结束时停止，而是会继续重复采样数据，直到手动停止训练。
    # dataset=dict(
    #     type='RepeatDataset',
    #     times=40000,
    #     dataset=dict(
    #         type=dataset_type,
    #         data_root=data_root,
    #         data_prefix=dict(
    #             img_path='images/training',
    #             seg_map_path='annotations/training'),
    #         pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,#num_workers 控制 DataLoader 读取数据的并行线程数。
    pin_memory=True,#加速数据拷贝到 GPU
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    nan_to_num=0)
test_evaluator = val_evaluator
