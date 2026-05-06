_base_ = ['./retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable.py']

# Fusion-weight ablation: increase detail branch contribution from 0.75 to 1.00.
model = dict(
    backbone=dict(
        fusion_alpha=1.0,
    ),
)

work_dir = './work_dirs_v3/retinanet_hybridmambadet_b_fpn_1x_visdrone_es_fusion10_stable'
