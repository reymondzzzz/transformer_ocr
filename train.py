from argparse import ArgumentParser
from pathlib import Path

import torch
from detector_utils.utils.other import load_module
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

import wandb
from ocr.utils.builders import build_lightning_module, build_callbacks_from_cfg
from ocr.utils.common import seed_everything_deterministic, get_checkpoint_callback

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--gpus', default=None, required=False, type=int, nargs='+')
    args = parser.parse_args()

    if not args.config.exists():
        assert False, f"Config not found: {args.config}"

    config = load_module(args.config)
    if config.seed is not None:
        seed_everything_deterministic(config.seed)
    if hasattr(config, 'mp_start_method'):
        torch.multiprocessing.set_start_method(config.mp_start_method)

    if args.gpus is not None:
        config.trainer_cfg['gpus'] = list(map(int, args.gpus))

    lightning_module = build_lightning_module(config.module_cfg)

    logger = WandbLogger(**config.wandb_cfg)
    logger.watch(lightning_module)
    if rank_zero_only.rank == 0:
        wandb.save(str(args.config))

    if 'callbacks' in config.trainer_cfg:
        config.trainer_cfg['callbacks'] = [
            build_callbacks_from_cfg(config)
            for config in config.trainer_cfg['callbacks']
        ]

    trainer = Trainer(
        logger=[logger],
        **config.trainer_cfg
    )

    trainer.fit(lightning_module)
    if rank_zero_only.rank == 0:
        maybe_cp_callback = get_checkpoint_callback(config.trainer_cfg['callbacks'])
        if maybe_cp_callback is not None:
            wandb.save(maybe_cp_callback.best_model_path)