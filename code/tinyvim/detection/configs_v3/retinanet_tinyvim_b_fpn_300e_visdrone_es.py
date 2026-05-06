_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/visdrone_detection_v2.py',
    '_base_/schedules/schedule_300e_earlystop.py',
    '_base_/default_runtime.py',
]

custom_imports = dict(imports=['model'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='TinyViM_B',
        fork_feat=True,
        init_cfg=dict(type='Pretrained', checkpoint='../../../weights/tinyvim/tinyvim_b_300e.pth')),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        num_classes=10,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300,
    ),
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
        min_delta=0.0005,
        patience=30,
        strict=False,
        check_finite=True,
    ),
]

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_300e_visdrone_es'
