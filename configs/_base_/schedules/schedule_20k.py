# optimizer，#SGD：噪声把你从鞍点推离，
# 更容易继续下降  η 学习率越大、m batch越小，抖动越大 
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='AdamW', lr=0.002, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.adapter': dict(lr_mult=20.0)  # 1e-4 * 50 = 5e-3（合理）
        }
    )
)
# learning policy,学习率调度器
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None,accumulative_counts=4)
# #accumulative_counts=4梯度累积


param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=100,
        by_epoch=True)
]
# training schedule for 20k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),

    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
   #interval=5, 控制台每5个iter打印
   #log_metric_by_epoch=True (tensorboard按epoch记录)
   #### `log_with_hierarchy=True` (分层命名)
    # TensorBoard中会以**分层结构**显示指标：
    # train/
    #   ├── loss
    #   ├── decode.loss_ce
    #   ├── aux.loss_ce
    #   └── acc_seg

    param_scheduler=dict(type='ParamSchedulerHook'),
    #遍历执行器的所有优化器参数调整策略（Parameter Scheduler）并逐个调用 step 方法更新优化器的参数

    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=3, max_keep_ckpts=3, save_best='mIoU'),
    #checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1500),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook', by_epoch=True))
    visualization=dict(
        type='SegVisualizationHook',
        draw=False))
