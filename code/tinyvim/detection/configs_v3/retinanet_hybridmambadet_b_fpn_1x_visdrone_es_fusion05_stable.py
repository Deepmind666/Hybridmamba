_base_ = ['./retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable.py']

# Fusion-weight ablation: reduce detail branch contribution from 0.75 to 0.50.
model = dict(
    backbone=dict(
        fusion_alpha=0.5,
    ),
)

work_dir = './work_dirs_v3/retinanet_hybridmambadet_b_fpn_1x_visdrone_es_fusion05_stable'
