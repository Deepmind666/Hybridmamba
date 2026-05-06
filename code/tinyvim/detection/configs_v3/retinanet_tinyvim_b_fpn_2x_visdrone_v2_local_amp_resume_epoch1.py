_base_ = [
    './retinanet_tinyvim_b_fpn_2x_visdrone_v2_local_amp.py',
]

load_from = '/mnt/c/mamba/artifacts/runs/visdrone_tinyvim_b_v2_local_ampfix2_gameturbo_20260421_1604/epoch_1.pth'
resume = True
