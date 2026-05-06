_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/visdrone_detection_v2.py',
    '_base_/schedules/schedule_2x.py',
    '_base_/default_runtime.py',
]

model = dict(
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

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001),
)

work_dir = './work_dirs_v3/retinanet_r50_fpn_2x_visdrone_sanity'
