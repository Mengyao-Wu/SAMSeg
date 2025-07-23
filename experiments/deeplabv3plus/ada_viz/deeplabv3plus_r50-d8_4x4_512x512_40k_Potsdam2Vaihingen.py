_base_ = [
    '../../../configs/_base_/datasets/pv_adaseg.py', '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderADABase',
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
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, reduction='none', loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5])
    ),
    # auxiliary_head=dict(
    #         type='FCNHead',
    #         in_channels=1024,
    #         in_index=2,
    #         channels=256,
    #         num_convs=1,
    #         concat_input=False,
    #         dropout_ratio=0.1,
    #         num_classes=6,
    #         norm_cfg=norm_cfg,
    #         align_corners=False,
    #         loss_decode=dict(type='CrossEntropyLoss', reduction='none',use_sigmoid=False, loss_weight=0.4, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5]),
    #         # loss_decode=dict(type='LovaszLoss',reduction='none', loss_weight=0.4, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5])
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

total_iters = 50000
check_iters = 50
sample_iters = 10000
ada_super_args = dict(
    # mode='RA',  # ['RA', 'PA']
    mode='TEST',  # ['RA', 'PA','TEST']
    pixels=40, ratio=0.022,
    radius=1,
    # sample_way='batch',  # ['total or batch']
    sample_way='total',  # ['total or batch']
    sample_num=(total_iters // sample_iters) - 1,
)
if ada_super_args['mode'] == 'RA':
    work_dir = './checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada_viz/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_Base_RA_{}/'.format(ada_super_args['ratio'])
elif ada_super_args['mode'] == 'PA':
    work_dir = './checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada_viz/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_Base_PA_{}/'.format(ada_super_args['pixels'])
elif ada_super_args['mode'] == 'TEST':
    # work_dir = './checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_test_{}_{}_only/'.format(ada_super_args['ratio'],ada_super_args['pixels'])
    work_dir = './checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada_viz/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_test_{}_{}_impurity/'.format(ada_super_args['ratio'],ada_super_args['pixels'])
else:
    raise ValueError('Not support mode :{}, must in [\'RA\' or \'PA\']'.format(ada_super_args['mode']))

# work_dir = './checkpoints/deeplabv3plus/Potsdam2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base_lov/'

checkpoint_config = dict(by_epoch=False, interval=check_iters)

ada_fixed_args = dict(
    default=2, al=2, kl=1,
    save_dir=f'{work_dir}', active_api='write', num_classes=model['decode_head']['num_classes'],
)  # 可覆写新增参数
ada_args = dict(**ada_fixed_args, **ada_super_args)
data = dict(samples_per_gpu=3, workers_per_gpu=3, train=dict(ada_args=dict(save_dir=work_dir, active_api='read')), ada=dict(ada_args=ada_args))  # 可覆写新增参数

evaluation = dict(interval=check_iters, metric=['mIoU', 'mFscore'], pre_eval=False)
ada_evaluation = dict(interval=sample_iters, metric=['mIoU', 'mFscore'], pre_eval=True)
# runner = None
runner = dict(type='IterBasedRunner', max_iters=total_iters)
find_unused_parameters = True
