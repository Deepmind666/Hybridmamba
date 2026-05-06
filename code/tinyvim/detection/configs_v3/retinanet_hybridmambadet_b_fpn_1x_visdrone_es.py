_base_ = ['./retinanet_hybridmambadet_b_fpn_1x_visdrone.py']

# Formal continuous run with early stopping for HybridMambaDet on VisDrone.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=5,
    ),
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        min_delta=0.0005,
        patience=30,
        strict=False,
        check_finite=True,
    ),
]
