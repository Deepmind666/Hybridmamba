_base_ = ['./retinanet_tinyvim_b_fpn_72e_aitodv2_stable.py']

# AI-TOD-v2 tiny-object protocol v1.
# Dataset stats in data/converted/aitodv2 show median boxes around 12x12 px
# and dense images with hundreds of instances. The default RetinaNet test cap
# of max_per_img=100 is therefore too restrictive for this benchmark.
model = dict(
    bbox_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.3,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            gpu_assign_thr=64,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=1000,
    ),
)

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_72e_aitodv2_tinyproto'
