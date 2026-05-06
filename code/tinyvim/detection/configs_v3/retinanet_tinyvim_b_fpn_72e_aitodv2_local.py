_base_ = ['./retinanet_tinyvim_b_fpn_120e_aitodv2_local.py']

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=72, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1500),
    dict(type='MultiStepLR', begin=0, end=72, by_epoch=True, milestones=[48, 66], gamma=0.1),
]

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        min_delta=0.002,
        patience=12,
        strict=False,
        check_finite=True,
    ),
]

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_72e_aitodv2_local'
