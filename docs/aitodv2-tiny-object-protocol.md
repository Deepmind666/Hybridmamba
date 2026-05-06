# AI-TOD-v2 Tiny-Object Protocol v1

This protocol is a controlled detector-side correction for AI-TOD-v2. It does
not change the backbone. It is intended to test whether the weak AP comes from
the HybridMambaDet idea or from a COCO-style RetinaNet protocol that is too
coarse for dense tiny objects.

## Dataset Evidence

Converted AI-TOD-v2 COCO annotation statistics:

- Train: 11214 images, 301494 boxes, 26.89 boxes/image.
- Val: 2804 images, 75082 boxes, 26.78 boxes/image.
- Median box size: 12 px width and 12 px height.
- Val density tail: 95th percentile 123 boxes/image, 99th percentile 253,
  maximum 815.
- Train density tail: 95th percentile 125 boxes/image, 99th percentile 278,
  maximum 1727.

## Config Changes

Relative to the current 72e AI-TOD-v2 RetinaNet configs:

- Anchor scale: `octave_base_scale=4` -> `2`, retaining P2 stride 4.
- Assignment: `pos_iou_thr=0.5/neg_iou_thr=0.4` -> `0.4/0.3`.
- Inference candidates: `nms_pre=1000` -> `3000`.
- Dense output cap: `max_per_img=100` -> `1000`.
- Score threshold: `0.05` -> `0.01`.
- NMS IoU: `0.5` -> `0.6`.

## Prepared Configs

- `code/tinyvim/detection/configs_v3/retinanet_tinyvim_b_fpn_72e_aitodv2_tinyproto.py`
- `code/tinyvim/detection/configs_v3/retinanet_hybridmamba_base_b_fpn_72e_aitodv2_tinyproto.py`
- `code/tinyvim/detection/configs_v3/retinanet_hybridmambadet_b_fpn_72e_aitodv2_fusion05_tinyproto.py`

## Interpretation Rule

If all backbones improve similarly, the prior low AP was mostly a detector
protocol issue. If TinyViM-B improves but HybridMambaDet remains behind, the
HybridMambaDet paper claim is weak for AI-TOD-v2. If HybridMambaDet closes the
gap or surpasses TinyViM-B under this corrected protocol, the project regains
paper potential.
