# Short VisDrone-V2 run to validate the BMVC-aligned Hybrid config before Fat formal.
_base_ = ['./retinanet_hybridmambadet_b_fpn_300e_visdrone_es_bs1_cpuassign.py']

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)

custom_hooks = []
