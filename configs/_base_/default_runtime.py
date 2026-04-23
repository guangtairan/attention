default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
#LoggerHook 整理好的信息打印到终端、写入文件、TensorBoard、WandB 等可视化工具
#log_processor 	收集训练过程中的指标（loss、accuracy等），并决定哪些要显示、如何汇总、多久更新一次

log_level = 'INFO'
load_from = None
resume = False
#开启 resume 时，会优先从 latest.pth 恢复；此时 load_from 通常被忽略。不建议同时使用。

tta_model = dict(type='SegTTAModel')
#测试时增强（TTA）通过生成多个增强版本的输入图像，
# 并对这些增强版本进行预测，然后将这些预测结果进行融合（例如取平均）
