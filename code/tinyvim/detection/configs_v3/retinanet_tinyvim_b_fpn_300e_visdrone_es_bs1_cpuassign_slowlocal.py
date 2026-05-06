_base_ = [
    './retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign.py',
]

# Keep the model and training protocol intact; only reduce host-side pressure.
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)

test_dataloader = val_dataloader

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=5,
    ),
)

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign_slowlocal'
