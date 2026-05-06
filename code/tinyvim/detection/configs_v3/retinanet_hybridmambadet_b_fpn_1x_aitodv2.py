_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/aitodv2_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py',
]

custom_imports = dict(imports=['model'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='HybridMambaDet_B',
        fork_feat=True,
        freq_split='avg3',
        detail_branch='dw_gate',
        detail_stages=(0, 1, 2),
        fusion_alpha=0.75,
        init_cfg=dict(type='Pretrained', checkpoint='../../../weights/tinyvim/tinyvim_b_300e.pth')),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
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

work_dir = './work_dirs_v3/retinanet_hybridmambadet_b_fpn_1x_aitodv2'
