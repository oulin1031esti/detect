# model settings
model = dict(
    type='TTFNet',
    pretrained="/project/train/src_repo/detect/premodels/mobilenet_v2-b0353104.pth",
    backbone=dict(
        type='MobileNetV2',
        out_indices=(1, 2, 4, 6),
        frozen_stages=-1,
        norm_eval=False
        ),
    neck=None,
    bbox_head=dict(
        type='TTFHeadv2',
        inplanes=(24, 32, 96, 320),
        fpn_outplane=32,
        asff_outplane=32,
        ssh_outplane=24,
        num_classes=2,
        wh_area_process='log',
        wh_agnostic=True,
        wh_gaussian=True,
        alpha=0.54,
        beta=0.54,
        hm_weight=1.,
        wh_weight=5.,
        max_objs=128,)
    )
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings
dataset_type = 'RatsDataset'
data_root = '/home/data/18/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 768), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file= '/project/train/rat_image_id.txt',
        img_prefix=data_root,
        pipeline=train_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_val2017.json',
    #     img_prefix=data_root + 'val2017/',
    #     pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_val2017.json',
    #     img_prefix=data_root + 'val2017/',
    #     pipeline=test_pipeline)
    )
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0004,
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[43, 47])
checkpoint_config = dict(interval=10, out_dir="/project/train/models/final/")
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', )
    ])
# yapf:enable
# runtime settings
total_epochs = 50
device_ids = range(1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/project/train/log/'
load_from = None
resume_from = None
workflow = [('train', 1)]
