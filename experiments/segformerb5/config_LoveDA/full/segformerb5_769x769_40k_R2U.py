_base_ = [
    '../../../../configs/_base_/datasets/ru_daseg.py', '../../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderUDAFull',
    pretrained=None,

    backbone=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/mit_b5.pth'),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.25, 1.0, 1.25, 1.0, 1.5, 1.25, 1.0])),
    # auxiliary_head=dict(
    #     type='SegformerHead',
    #     in_channels=[64, 128, 320, 576],
    #     in_index=[0, 1, 2, 3],
    #     channels=256,
    #     dropout_ratio=0.1,
    #     num_classes=7,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.25, 1.0, 1.25, 1.0, 1.5, 1.25, 1.0])),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)),
)

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
# optimizer setting
optimizer = dict(
    backbone=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
    decode_head=dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })),
)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

work_dir = './checkpoints/deeplabv3plus/LoveDA_R2U_results/segformerb5_769x769_40k_full_test/'

data = dict(samples_per_gpu=3, workers_per_gpu=3,)
# total_iters = 40000
checkpoint_config = dict(by_epoch=False, interval=2500)
evaluation = dict(interval=2500, metric=['mIoU', 'mFscore'], pre_eval=True)
# runner = None
runner = dict(type='IterBasedRunner', max_iters=40000)
find_unused_parameters = True
