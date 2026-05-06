_base_ = [
    './retinanet_tinyvim_b_fpn_2x_visdrone_v2.py',
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    pin_memory=False,
)

test_dataloader = val_dataloader

model = dict(
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            gpu_assign_thr=32,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=2)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.05),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=5),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_2x_visdrone_v2_local_fp32'
