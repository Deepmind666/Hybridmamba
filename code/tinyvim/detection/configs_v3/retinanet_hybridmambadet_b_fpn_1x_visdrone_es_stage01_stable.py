_base_ = ['./retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable.py']

# Local next-step ablation: keep the detail branch but restrict it to shallow stages.
model = dict(
    backbone=dict(
        detail_stages=(0, 1),
    ),
)

work_dir = './work_dirs_v3/retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stage01_stable'
