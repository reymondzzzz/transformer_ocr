seed = 42
gpus = [0, 1]
batch_size = 128
epochs = 1000
img_side_size = 256
sequence_size = 14
num_workers = 12 // len(gpus)

vocab = '0123456789abcdefghijklmnopqrstuvwxyzäüö'
letters = ['pad', 'sos'] + list(vocab) + ['eos']
hidden_features = 512
emb_size = 256
train_dataset_len = 30138
debug = False

trainer_cfg = dict(
    gpus=gpus,
    max_epochs=epochs,
    callbacks=[
        dict(type='LearningRateMonitor', logging_interval='step'),
        dict(type='ModelCheckpoint', save_top_k=5, verbose=True, mode='max',
             monitor='val_accuracy', dirpath='./results/attention/',
             filename='{epoch:02d}_{val_accuracy:.4f}')
    ],
    resume_from_checkpoint=None,
    benchmark=True,
    deterministic=True,
    terminate_on_nan=True,
    distributed_backend='ddp',
    precision=16,
    sync_batchnorm=True,
)

wandb_cfg = dict(
    name=f'{__file__.split("/")[-1].replace(".py", "")}_{img_side_size}_{batch_size}_ep{epochs}',
    project='ocr'
)

backbone_cfg = dict(
    type='RepVGG_B1',
    # deploy=True
)

decoder_cfg = dict(
    type='BahdanauAttnDecoderRNN',
    hidden_features=hidden_features,
    embed_size=emb_size,
    vocab=letters
)

encoder_cfg = dict(
    type='AttentionEncoder',
    hidden_features=hidden_features,
    feature_x=8,
    feature_y=8,
    # vocab=letters
)

loss_cfgs = [
    dict(type='CrossEntropyLoss',
         name='cross_entropy',
         ignore_index=0  # pad class
    )
]


metric_cfgs = [
    dict(type='Accuracy', name='val_accuracy'),
    dict(type='PhonemeErrorRate', name='val_PER', letters=letters),
]

killer_scale_min, killer_scale_max = 20 / img_side_size, 30 / img_side_size
killer_transforms = dict(
    type='Compose', transforms=[
        dict(type='OneOf', transforms=[
            # dict(type='OneOf', transforms=[
            #     dict(type='GaussianBlur', blur_limit=(17, 21), p=1),
            #     dict(type='MedianBlur', blur_limit=(17, 21), p=1),
            #     dict(type='MotionBlur', blur_limit=(19, 27), p=1),
            # ]),
            dict(type='OneOf', transforms=[
                dict(type='Downscale', scale_min=killer_scale_min, scale_max=killer_scale_max, interpolation=0, p=1.),
                dict(type='Downscale', scale_min=killer_scale_min, scale_max=killer_scale_max, interpolation=1, p=1.),
                dict(type='Downscale', scale_min=killer_scale_min, scale_max=killer_scale_max, interpolation=4, p=1.)
            ]),
            # dict(type='Compose', transforms=[
            #     dict(type='GaussianBlur', sigma_limit=(7, 11), p=1),
            #     dict(type='ImageCompression', quality_lower=95, quality_upper=100, p=1),
            # ])
        ], p=1),
        dict(type='NullifyText')
    ], p=0.1
)

scale_min, scale_max = 0.7, 0.99
real_transforms = dict(type='Compose', transforms=[
    # TODO 3d rotate
    dict(type='OneOf', transforms=[
        dict(type='Downscale', scale_min=scale_min, scale_max=scale_max, interpolation=0, p=1.),
        dict(type='Downscale', scale_min=scale_min, scale_max=scale_max, interpolation=1, p=1.),
        dict(type='Downscale', scale_min=scale_min, scale_max=scale_max, interpolation=4, p=1.)
    ], p=0.7),
    dict(type='RandomBrightnessContrast', brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2),
         p=0.5),
    dict(type='RGBShift', r_shift_limit=(10, 20), g_shift_limit=(10, 20), b_shift_limit=(10, 20),
         p=0.7),
    dict(type='OneOf', transforms=[
        dict(type='MotionBlur', p=1.),
        dict(type='Blur', blur_limit=3, p=1.),
        dict(type='MedianBlur', blur_limit=3, p=1.)
    ], p=0.2),
    dict(type='SeriesTransformation',
         series_size=0,  # series_size,
         pitch_angle=0.05,
         roll_angle=0.05,
         scale=(-0.1, -0.1), dx=0, dy=0,
         yaw_angle=5, p=1),
], p=0.9)

train_transforms_cfg = dict(
    real_dataset=dict(
        type='Compose', transforms=[
            dict(type='LongestMaxSize', max_size=max(img_side_size, img_side_size)),
            dict(type='PadIfNeeded', min_width=img_side_size, min_height=img_side_size, value=(128, 128, 128),
                 border_mode=0),
            dict(type='OneOf', transforms=[
                killer_transforms,
                real_transforms
            ], p=1),
            dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
            dict(type='Tokenizer', vocab=letters, seq_size=sequence_size),
            dict(type='ToTensorV2')
        ]),
    fake_dataset=dict(
        type='Compose', transforms=[
            dict(type='LongestMaxSize', max_size=max(img_side_size, img_side_size)),
            dict(type='ToGray', p=0.5),
            dict(type='IAAAdditiveGaussianNoise', loc=0, scale=(64.0, 64.0), p=0.5),
            dict(type='IAASharpen', alpha=(0, 1.0), lightness=(0.5, 4.0), p=0.5),
            dict(type='ChannelShuffle', p=0.5),
            dict(type='MotionBlur', blur_limit=5, p=0.7),
            dict(type='PadIfNeeded', min_width=img_side_size, min_height=img_side_size, value=(128, 128, 128),
                 border_mode=0),
            dict(type='SeriesTransformation',
                 series_size=0,  # series_size,
                 pitch_angle=0.05,
                 roll_angle=0.05,
                 scale=0.0, dx=0, dy=0,
                 yaw_angle=5, p=1),
            dict(type='ShiftScaleRotate', shift_limit=0.02, scale_limit=0.02, rotate_limit=5,
                 border_mode=0, value=(128, 128, 128), p=1),
            dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
            dict(type='Tokenizer', vocab=letters, seq_size=sequence_size),
            dict(type='ToTensorV2')
        ]),
    empty_dataset=dict(
        type='Compose', transforms=[
            dict(type='LongestMaxSize', max_size=max(img_side_size, img_side_size)),
            dict(type='PadIfNeeded', min_width=img_side_size, min_height=img_side_size, value=(128, 128, 128),
                 border_mode=0),
            real_transforms,
            dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
            dict(type='Tokenizer', vocab=letters, seq_size=sequence_size),
            dict(type='ToTensorV2')
        ]),
)

val_transforms_cfg = dict(
    type='Compose', transforms=[
        dict(type='LongestMaxSize', max_size=max(img_side_size, img_side_size)),
        dict(type='PadIfNeeded', min_width=img_side_size, min_height=img_side_size, value=(128, 128, 128),
             border_mode=0),
        dict(type='Tokenizer', vocab=letters, seq_size=sequence_size),
        dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
        dict(type='ToTensorV2')
    ])

train_dataset_cfg = [
    dict(
        type='RealDataset',
        dataset_path='/home/kstarkov/ml/datasets/lpr4_images',
        subset='train',
        vocab=vocab,
        debug=debug,
        lines_allowed=[1, 2],
        name='real_dataset'
    ),
    dict(
        type='FakeDataset',
        capacity=4637,  # len(real_dataset) * 0.2
        name='fake_dataset',
        debug=debug,
        generator_config=dict(
            lpr_resources='/home/kstarkov/t1s/tech1lpr/lpr_resources',
            most_popular_templates=dict(
                ro_type1_subtype1_lines1=0.04,
                ro_type1_subtype2_lines1=0.04,
                ro_type1_subtype3_lines1=0.04,
                ro_type2_subtype1_lines1=0.04,
                ro_type3_subtype1_lines1=0.04,

                ru_type5_subtype1_lines1=0.1333,
                ru_type5_subtype2_lines1=0.1333,
                ru_type5_subtype3_lines1=0.1333,
                ru_type6_subtype1_lines1=0.1,
                ru_type7_subtype1_lines1=0.1,
            ),
            most_popular_letters={
                'x': .25,
                'y': .25,
                '7': .3,
                'z': .25,
            }
        )
    ),
    dict(
        type='EmptyDataset',
        debug=debug,
        root_path='/home/kstarkov/ml/datasets/lpr4_images',
        capacity=2318,  # len(real_dataset) * 0.1,
        name='empty_dataset'
    )
]

val_dataset_cfg = dict(
    type='RealDataset',
    dataset_path='/home/kstarkov/ml/datasets/lpr4_images',
    subset='val',
    vocab=vocab,
    lines_allowed=[1, 2]
)

train_dataloader_cfg = dict(
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

val_dataloader_cfg = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

optimizer_cfg = dict(
    type='RangerAdaBelief',
    lr=1e-3 * len(gpus)
)
scheduler_cfg = dict(
    type='CyclicLR',
    base_lr=1e-3 * len(gpus),
    max_lr=1e-2 * len(gpus),
    step_size_up=int(train_dataset_len // batch_size * (epochs * 0.1)),
    mode='triangular2',
    cycle_momentum=False,
)
scheduler_update_params = dict(
    interval='step',
    frequency=1
)

module_cfg = dict(
    type='TransformerOCRPLModule',
    vocab=letters,
    sequence_size=sequence_size,
    backbone_cfg=backbone_cfg,
    # head_cfg=head_cfg,
    decoder_cfg=decoder_cfg,
    encoder_cfg=encoder_cfg,
    # loss_head_cfg=loss_head_cfg,
    loss_cfgs=loss_cfgs,
    metric_cfgs=metric_cfgs,
    train_transforms_cfg=train_transforms_cfg,
    val_transforms_cfg=val_transforms_cfg,
    train_dataset_cfg=train_dataset_cfg,
    val_dataset_cfg=val_dataset_cfg,
    train_dataloader_cfg=train_dataloader_cfg,
    val_dataloader_cfg=val_dataloader_cfg,
    optimizer_cfg=optimizer_cfg,
    scheduler_cfg=scheduler_cfg,
    scheduler_update_params=scheduler_update_params
)
