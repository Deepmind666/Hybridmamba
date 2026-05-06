_base_ = ['./retinanet_tinyvim_b_fpn_120e_aitodv2_local.py']

# Ultra-safe local profile to reduce desktop lag/stall risk.
image_scale = (960, 544)

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

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(pipeline=test_pipeline, test_mode=True),
)

test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        min_delta=0.003,
        patience=8,
        strict=False,
        check_finite=True,
    ),
]

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_120e_aitodv2_local_ultrasafe'
