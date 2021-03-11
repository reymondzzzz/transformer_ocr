import torch
from argparse import ArgumentParser
from detector_utils.utils.other import load_module
from pathlib import Path
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
    args = parser.parse_args()

    if not args.config.exists():
        assert False, f"Config not found: {args.config}"

    config = load_module(args.config)
    if args.gpus is not None:
        config.trainer_cfg['gpus'] = list(map(int, args.gpus))

    lightning_module = build_lightning_module(config.module_cfg)

    input_sample = torch.randn((128, 3, 256, 256))
    lightning_module.to_onnx('model.onnx', input_sample, input_names=['input'],
                             output_names=['output'],
                             # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                             custom_opsets={'CustomPlugin': 2},
                             dynamic_axes={
                                 'input': {0: 'bs'},
                                 'output': {0: 'bs'}
                             },
                             opset_version=11)
