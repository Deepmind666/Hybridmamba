_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/aitodv2_detection.py',
    '_base_/default_runtime.py',
]

custom_imports = dict(imports=['model'], allow_failed_imports=False)

image_scale = (1333, 800)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='HybridMambaDet_B',
        fork_feat=True,
        freq_split='avg3',
        detail_branch='dw_gate',
        detail_stages=(0, 1, 2),
        fusion_alpha=0.75,
        init_cfg=dict(type='Pretrained', checkpoint='../../../weights/tinyvim/tinyvim_b_300e.pth'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
    ),
    bbox_head=dict(
        num_classes=8,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            gpu_assign_thr=64,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img='images/train/images/'),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(
        ann_file='annotations/val_coco.json',
        data_prefix=dict(img='images/val/images/'),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='../../../data/converted/aitodv2/annotations/val_coco.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
)

test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1500),
    dict(type='MultiStepLR', begin=0, end=120, by_epoch=True, milestones=[80, 110], gamma=0.1),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=5,
    ),
)

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

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

work_dir = './work_dirs_v3/retinanet_hybridmambadet_b_fpn_120e_aitodv2_stable'
