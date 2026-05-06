_base_ = ['./retinanet_hybridmambadet_b_fpn_1x_visdrone.py']

# Stable training profile validated by recent long-running local experiments.
train_cfg = dict(max_epochs=120)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(pipeline=test_pipeline),
)
test_dataloader = val_dataloader

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
        min_delta=0.0005,
        patience=20,
        strict=False,
        check_finite=True,
    ),
]

work_dir = './work_dirs_v3/retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable'
