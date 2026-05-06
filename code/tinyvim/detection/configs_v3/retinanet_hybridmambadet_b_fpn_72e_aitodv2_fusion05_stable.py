_base_ = ['./retinanet_hybridmambadet_b_fpn_72e_aitodv2_stable.py']

model = dict(
    backbone=dict(
        fusion_alpha=0.5,
    ),
)

work_dir = './work_dirs_v3/retinanet_hybridmambadet_b_fpn_72e_aitodv2_fusion05_stable'
