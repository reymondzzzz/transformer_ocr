import sys
from typing import Tuple

import albumentations as Albumentations
import albumentations.pytorch as AlbumentationsPytorch
import mask_the_face as MaskTransforms
import pretrainedmodels
import pytorch_lightning.callbacks as LightningCallbacks
import pytorch_loss as PytorchExtraLosses
import timm
import torch
import torch.nn as TorchNNModules
import torch.optim as OptimizerLib
import torch.optim.lr_scheduler as LRSchedulerLib
import pytorch_lightning.metrics as PLMetrics

import ocr.decoder as Decoders
import ocr.decoder_head as DecoderHeades
import ocr.datasets as Datasets
import ocr.loss as Losses
import ocr.metrics as CustomMetrics
import ocr.modelling.backbones as Backbones
import ocr.modelling.heads as Heads
import ocr.transforms as Transforms

__all__ = [
    'build_lightning_module',
    'build_backbone_from_cfg',
    'build_head_from_cfg',
    'build_transform_from_cfg',
    'build_dataset_from_cfg',
    'build_loss_from_cfg',
    'build_metric_from_cfg',
    'build_optimizer_from_cfg',
    'build_lr_scheduler_from_cfg',
    'build_callbacks_from_cfg'
]


def _base_transform_from_cfg(config, modules_to_find):
    assert isinstance(config, dict) and 'type' in config, f'Check config type validity: {config}'

    args = config.copy()
    obj_type_name = args.pop('type')

    real_type = None
    for module in modules_to_find:
        if not hasattr(module, obj_type_name):
            continue
        real_type = getattr(module, obj_type_name)
        if real_type:
            break

    assert real_type is not None, f'{obj_type_name} is not registered type in any modules: {modules_to_find}'
    return real_type(**args)


def build_lightning_module(config):
    import ocr.modules as Modules
    return _base_transform_from_cfg(config, [Modules])


def build_backbone_from_cfg(config) -> Tuple[torch.nn.Module, int]:
    args = config.copy()
    backbone_type_name = args.pop('type')

    if hasattr(Backbones, backbone_type_name):
        backbone = getattr(Backbones, backbone_type_name)(**args)
        output_channels = backbone.output_channels
    elif backbone_type_name in pretrainedmodels.__dict__:
        backbone = pretrainedmodels.__dict__[backbone_type_name](**args)
        if 'squeezenet' in backbone_type_name:
            backbone = backbone.features
            output_channels = 512
        else:
            backbone.forward = backbone.features
            output_channels = backbone.last_linear.in_features
    elif backbone_type_name in timm.list_models():
        backbone = timm.create_model(backbone_type_name, **args)
        backbone.forward = backbone.forward_features
        output_channels = backbone.classifier.in_features
    else:
        assert False, f'{backbone_type_name} not found in backbones factory'

    return backbone, output_channels


def build_head_from_cfg(input_channels: int, config):
    config['input_channels'] = input_channels
    return _base_transform_from_cfg(config, [Heads])


def build_transform_from_cfg(config):
    def _builder(cfg):
        modules = [Albumentations, AlbumentationsPytorch, Transforms, MaskTransforms]
        try:
            sys.path.append('./face_reid/external/dddfa_v2_dssl')
            import face_reid.external.dddfa_v2_dssl.tddfa_v2.transform as TddfaTranform
            modules.append(TddfaTranform)
        except:
            pass

        if 'transforms' in cfg:
            cfg['transforms'] = [
                _builder(transform_cfg) for transform_cfg in cfg['transforms']
            ]

        return _base_transform_from_cfg(cfg, modules)

    return _builder(config)


def build_dataset_from_cfg(transforms, config):
    config['transforms'] = transforms
    return _base_transform_from_cfg(config, [Datasets])


def build_loss_from_cfg(config):
    if config['type'] == 'MixedLoss':
        losses = []
        for loss_cfg in config['losses']:
            losses.append(_base_transform_from_cfg(loss_cfg, [Losses, TorchNNModules, PytorchExtraLosses]))
        return Losses.MixedLoss(losses=losses)
    else:
        return _base_transform_from_cfg(config, [Losses, TorchNNModules, PytorchExtraLosses])


def build_decoder_from_cfg(config):
    return _base_transform_from_cfg(config, [Decoders])


def build_decoder_head_from_cfg(config):
    return _base_transform_from_cfg(config, [DecoderHeades])


def build_metric_from_cfg(config):
    return _base_transform_from_cfg(config, [PLMetrics, CustomMetrics])


def build_optimizer_from_cfg(params, config):
    modules = [OptimizerLib]
    try:
        import adabelief_pytorch
        modules.append(adabelief_pytorch)
    except ImportError:
        pass

    try:
        import ranger_adabelief
        modules.append(ranger_adabelief)
    except ImportError:
        pass

    try:
        import ranger
        modules.append(ranger)
    except ImportError:
        pass

    config['params'] = params
    return _base_transform_from_cfg(config, modules)


def build_lr_scheduler_from_cfg(optimizer, config):
    config['optimizer'] = optimizer
    return _base_transform_from_cfg(config, [LRSchedulerLib])


def build_callbacks_from_cfg(config):
    return _base_transform_from_cfg(config, [LightningCallbacks])
