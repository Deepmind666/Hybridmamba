_base_ = ['./retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign.py']

# First-8 recovery under reduced load to avoid driver-not-ready instability.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
custom_hooks = []

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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign_first8_stable'
