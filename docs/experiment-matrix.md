# Experiment Matrix

## Mainline VisDrone

| Role | Config | Detector | Target Metric |
| --- | --- | --- | --- |
| Baseline | `code/tinyvim/detection/configs_v3/retinanet_tinyvim_b_fpn_1x_visdrone.py` | RetinaNet + FPN | AP, AP_S |
| Low-frequency baseline | `code/tinyvim/detection/configs_v3/retinanet_hybridmamba_base_b_fpn_1x_visdrone.py` | RetinaNet + FPN | AP, AP_S |
| Final model | `code/tinyvim/detection/configs_v3/retinanet_hybridmambadet_b_fpn_1x_visdrone.py` | RetinaNet + FPN | AP, AP_S, throughput |

## VisDrone V2 Protocol

| Role | Config | Detector | Purpose |
| --- | --- | --- | --- |
| Sanity detector | `code/tinyvim/detection/configs_v3/retinanet_r50_fpn_2x_visdrone_sanity.py` | RetinaNet + FPN | check whether the stronger protocol yields non-pathological VisDrone performance |
| Official tiny hybrid Mamba reproduction | `code/tinyvim/detection/configs_v3/retinanet_tinyvim_b_fpn_2x_visdrone_v2.py` | RetinaNet + FPN | re-establish a trustworthy TinyViM baseline before tuning our method |

## Transfer To AI-TOD-v2

| Role | Config | Detector | Target Metric |
| --- | --- | --- | --- |
| Baseline | `code/tinyvim/detection/configs_v3/retinanet_tinyvim_b_fpn_1x_aitodv2.py` | RetinaNet + FPN | AP, AP_S |
| Final model | `code/tinyvim/detection/configs_v3/retinanet_hybridmambadet_b_fpn_1x_aitodv2.py` | RetinaNet + FPN | AP, AP_S |

## Supplemental DOTA-HBB

| Role | Config | Detector | Target Metric |
| --- | --- | --- | --- |
| Baseline | `code/tinyvim/detection/configs_v3/retinanet_tinyvim_b_fpn_1x_dotahbb.py` | RetinaNet + FPN | AP |
| Final model | `code/tinyvim/detection/configs_v3/retinanet_hybridmambadet_b_fpn_1x_dotahbb.py` | RetinaNet + FPN | AP |

## Ablation Axes

1. `freq_split`: `avg3`, `avg5`
2. `detail_branch`: `none`, `dw_gate`
3. `detail_stages`: shallow only, shallow plus middle, all Mamba stages
4. `fusion_alpha`: sweep around `0.25`, `0.5`, `0.75`, `1.0`
