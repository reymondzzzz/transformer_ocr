seed = 42
gpus = [0]
batch_size = 128
epochs = 1000
img_side_size = 128
sequence_size = 14
num_workers = 12 // len(gpus)

vocab = '0123456789abcdefghijklmnopqrstuvwxyzäüö'
letters = ['pad', 'sos'] + list(vocab) + ['eos']
hidden_features = 128
train_dataset_len = 23183

trainer_cfg = dict(
    gpus=gpus,
    max_epochs=epochs,
    callbacks=[
        dict(type='LearningRateMonitor', logging_interval='step'),
        dict(type='ModelCheckpoint', save_top_k=5, verbose=True, mode='max',
             monitor='val_accuracy', dirpath='./results/',
             filename='{epoch:02d}_{val_accuracy:.2f}')
    ],
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
    type='RepVGG_A2',
)

head_cfg = dict(
    type='ConvHead',
    output_channels=hidden_features//4
)

decoder_cfg = dict(
    type='TransformerDecoder',
    hidden_features=hidden_features,
    vocab=letters
)

loss_cfgs = [
    dict(type='CrossEntropyLoss',
         ignore_index=0,
         name='cross_entropy')
]

metric_cfgs = [
    dict(type='Accuracy', name='val_accuracy'),
    dict(type='PhonemeErrorRate', name='val_PER', letters=letters),
]

# scale_min, scale_max = 0.7, 0.99
train_transforms_cfg = dict(
    type='Compose', transforms=[
        dict(type='LongestMaxSize', max_size=max(img_side_size, img_side_size)),
        dict(type='PadIfNeeded', min_width=img_side_size, min_height=img_side_size, value=(128, 128, 128),
             border_mode=0),
        # dict(type='OneOf', transforms=[
        #     dict(type='Downscale', scale_min=scale_min, scale_max=scale_max, interpolation=0, p=1.),
        #     dict(type='Downscale', scale_min=scale_min, scale_max=scale_max, interpolation=1, p=1.),
        #     dict(type='Downscale', scale_min=scale_min, scale_max=scale_max, interpolation=4, p=1.)
        # ], p=0.7),
        # dict(type='RandomBrightnessContrast', brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        # dict(type='RGBShift', r_shift_limit=(10, 20), g_shift_limit=(10, 20), b_shift_limit=(10, 20), p=0.7),
        # dict(type='OneOf', transforms=[
        #     dict(type='MotionBlur', p=1.),
        #     dict(type='Blur', blur_limit=3, p=1.),
        #     dict(type='MedianBlur', blur_limit=3, p=1.)
        # ], p=0.2),
        # dict(type='HueSaturationValue', p=0.3),
        dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
        dict(type='Tokenizer', vocab=letters, seq_size=sequence_size),
        dict(type='ToTensorV2')
    ])

val_transforms_cfg = dict(
    type='Compose', transforms=[
        dict(type='LongestMaxSize', max_size=max(img_side_size, img_side_size)),
        dict(type='PadIfNeeded', min_width=img_side_size, min_height=img_side_size, value=(128, 128, 128),
             border_mode=0),
        dict(type='Tokenizer', vocab=letters, seq_size=sequence_size),
        dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
        dict(type='ToTensorV2')
    ])
train_dataset_cfg = dict(
    type='RealDataset',
    dataset_path='/home/kstarkov/ml/datasets/lpr4_images',
    subset='train',
    vocab=vocab,
    debug=False,
    lines_allowed=[1, 2]
)

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
    lr=1e-4# * len(gpus)
)
scheduler_cfg = dict(
    type='CyclicLR',
    base_lr=1e-4 * len(gpus),
    max_lr=1e-3 * len(gpus),
    step_size_up=int(train_dataset_len // batch_size * (epochs * 0.1)),
    mode='triangular2',
    cycle_momentum=False,
)
# scheduler_cfg = dict(
#     type='ReduceLROnPlateau',
#     mode='min'
# )
scheduler_update_params = dict(
    interval='step',
    frequency=1
)

module_cfg = dict(
    type='TransformerOCRPLModule',
    vocab=letters,
    sequence_size=sequence_size,
    backbone_cfg=backbone_cfg,
    head_cfg=head_cfg,
    decoder_cfg=decoder_cfg,
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
