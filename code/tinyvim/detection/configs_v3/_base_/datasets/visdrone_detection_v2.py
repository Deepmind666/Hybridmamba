dataset_type = 'CocoDataset'
ann_root = '../../../data/converted/visdrone/'
img_root = '../../../data/visdrone/'

metainfo = dict(
    classes=(
        'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
        'tricycle', 'awning-tricycle', 'bus', 'motor',
    ),
)

backend_args = None

image_scale = (1600, 960)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=ann_root,
        metainfo=metainfo,
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img=img_root + 'train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=ann_root,
        metainfo=metainfo,
        ann_file='annotations/val_coco.json',
        data_prefix=dict(img=img_root + 'val/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_root + 'annotations/val_coco.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator
