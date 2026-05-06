_base_ = ['./retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable.py']

# OOM-safe fusion ablation:
# keep the same backbone setting as fusion10, but force the assigner toward
# CPU for most VisDrone samples so the Fat run can finish.
model = dict(
    backbone=dict(
        fusion_alpha=1.0,
    ),
    train_cfg=dict(
        assigner=dict(
            gpu_assign_thr=1,
        ),
    ),
)

work_dir = './work_dirs_v3/retinanet_hybridmambadet_b_fpn_1x_visdrone_es_fusion10_cpuassign_stable'
