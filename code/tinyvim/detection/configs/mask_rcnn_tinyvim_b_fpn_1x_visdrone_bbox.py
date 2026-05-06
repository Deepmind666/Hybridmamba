_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/visdrone_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py',
]

# TinyViM paper-aligned downstream detection protocol:
# TinyViM backbone + FPN + the Mask R-CNN two-stage detection branch.
# VisDrone2019-DET provides bounding boxes but no instance masks, so the mask
# branch is disabled and evaluation reports bbox AP only.
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='TinyViM_B',
        fork_feat=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../../../weights/tinyvim/tinyvim_b_300e.pth')),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_roi_extractor=None,
        mask_head=None),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=500)))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=12)

work_dir = './work_dirs/mask_rcnn_tinyvim_b_fpn_1x_visdrone_bbox'
