_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/visdrone_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py',
]

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='HybridMambaDet_B',
        fork_feat=True,
        freq_split='avg3',
        detail_branch='dw_gate',
        detail_stages=(0, 1, 2),
        fusion_alpha=0.75,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../../../weights/tinyvim/tinyvim_b_300e.pth')),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(num_classes=10),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=12)

work_dir = './work_dirs/retinanet_hybridmambadet_b_fpn_1x_visdrone'
