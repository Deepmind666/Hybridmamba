_base_ = ['./retinanet_tinyvim_b_fpn_72e_aitodv2_stable.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='HybridMamba_Base_B',
        fork_feat=True,
        freq_split='avg3',
        init_cfg=dict(type='Pretrained', checkpoint='../../../weights/tinyvim/tinyvim_b_300e.pth'),
    ),
)

work_dir = './work_dirs_v3/retinanet_hybridmamba_base_b_fpn_72e_aitodv2_stable'
