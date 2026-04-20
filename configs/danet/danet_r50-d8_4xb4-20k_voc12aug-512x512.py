_base_ = [
    '../_base_/models/danet_r50-d8.py',
    '../_base_/datasets/forest.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

#_base_支持继承多个文件，将同时获得这多个文件中的所有字段，
# 但是要求继承的多个文件中没有相同名称的字段，否则会报错
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))


#load_from、vis_backends、visualizer与model平级,  不需要加到model里面
#test_dataloader 控制如何加载和预处理测试数据  不需要加到model
#test_cfg 控制如何进行测试/推理以及如何处理模型输出，例如测试时的 NMS、推理模式、评估方法等  需要加到model


load_from = r'E:\Desktop\mmsegmentation-1.2.2\danet_r50-d8_512x512_20k_voc12aug_20200618_070026-9e9e3ab3.pth'


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)



# # ===== 可选：配置日志和可视化频率 =====
# default_hooks = dict(
#     logger=dict(type='LoggerHook', interval=50),
#     checkpoint=dict(
#         type='CheckpointHook',
#         by_epoch=False,
#         interval=2000,
#         save_best='mIoU'
#     ),
#     visualization=dict(
#         type='SegVisualizationHook',
#         draw=True,
#         interval=500
#     )
# )