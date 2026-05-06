_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/dotahbb_detection.py',
    '_base_/schedules/schedule_1x.py',
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
        num_classes=15,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
)

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_1x_dotahbb'
