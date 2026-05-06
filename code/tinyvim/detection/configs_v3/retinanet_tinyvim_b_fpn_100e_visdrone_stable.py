_base_ = ['./retinanet_tinyvim_b_fpn_1x_visdrone_es_stable.py']

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=True, milestones=[67, 92], gamma=0.1),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=5,
    ),
)

custom_hooks = []

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_100e_visdrone_stable'
