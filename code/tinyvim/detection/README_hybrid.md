# HybridMambaDet Notes

This repository keeps the upstream TinyViM detection tree intact and adds new backbones through `model/hybridmamba.py`.

## Added Backbones

- `HybridMamba_Base_B`
- `HybridMambaDet_B`

## Added Configs

- `configs/retinanet_tinyvim_b_fpn_1x_visdrone.py`
- `configs/retinanet_hybridmamba_base_b_fpn_1x_visdrone.py`
- `configs/retinanet_hybridmambadet_b_fpn_1x_visdrone.py`
- `configs/retinanet_tinyvim_b_fpn_1x_aitodv2.py`
- `configs/retinanet_hybridmambadet_b_fpn_1x_aitodv2.py`
- `configs/retinanet_tinyvim_b_fpn_1x_dotahbb.py`
- `configs/retinanet_hybridmambadet_b_fpn_1x_dotahbb.py`

## Dataset Assumptions

Converted annotations are expected under:

- `../../../data/converted/visdrone/`
- `../../../data/converted/aitodv2/`
- `../../../data/converted/dota_hbb/`

## Checkpoint Assumption

The TinyViM B checkpoint is expected at:

- `../../../weights/tinyvim/tinyvim_b_300e.pth`
