_base_ = [
    '../../../configs/_base_/datasets/ru_daseg.py', '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderUDAFull',
    pretrained='open-mmlab://resnet50_v1c',

    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.25, 1.0, 1.25, 1.0, 1.5, 1.25, 1.0])
    ),
    # auxiliary_head=dict(
    #         type='FCNHead',
    #         in_channels=1024,
    #         in_index=2,
    #         channels=256,
    #         num_convs=1,
    #         concat_input=False,
    #         dropout_ratio=0.1,
    #         num_classes=7,
    #         norm_cfg=norm_cfg,
    #         align_corners=False,
    #         loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[1.25, 1.0, 1.25, 1.0, 1.5, 1.25, 1.0]),
    #         # loss_decode=dict(type='LovaszLoss',reduction='none', class_weight=[1.25, 1.0, 1.25, 1.0, 1.5, 1.25, 1.0]),
    # ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)

# optimizer setting
optimizer = dict(
    backbone=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    decode_head=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005),
)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

work_dir = './checkpoints/deeplabv3plus/LoveDA_U2R_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full_test/'

data = dict(samples_per_gpu=3, workers_per_gpu=3,
            train=dict(
                img_dir='Train/Urban/images_png',
                ann_dir='Train/Urban/masks_png',
                split='TrainUrban.txt',
                B_img_dir='Train/Rural/images_png',
                B_ann_dir='Train/Rural/masks_png',
                B_split='TrainRural.txt',
                B_coco_mask_dir='Train/Rural/auto_mask_dir',
            ),
            # target domain for validation
            val=dict(
                img_dir='Val/Rural/images_png',
                ann_dir='Val/Rural/masks_png',
                split='ValRural.txt'),
            test=dict(
                # img_dir='Test/Rural/images_png',
                # ann_dir='Test/Rural/masks_png',
                # split='TestRural.txt'),
                img_dir='Val/Rural/images_png',
                ann_dir='Val/Rural/masks_png',
                split='ValRural.txt'),
            )
# total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=2500)
evaluation = dict(interval=2500, metric=['mIoU', 'mFscore'], pre_eval=True)
# runner = None
runner = dict(type='IterBasedRunner', max_iters=40000)
find_unused_parameters = True
