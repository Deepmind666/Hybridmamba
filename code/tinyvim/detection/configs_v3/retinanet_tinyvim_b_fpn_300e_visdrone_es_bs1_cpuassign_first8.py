_base_ = ['./retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign.py']

# Recovery run for missing early validation logs: strictly first 8 epochs.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)

# Disable early-stop for deterministic first-8 reconstruction.
custom_hooks = []

work_dir = './work_dirs_v3/retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign_first8'
