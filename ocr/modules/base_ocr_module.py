from copy import deepcopy
from typing import Dict, Any, Optional, List

import pytorch_lightning as pl

__all__ = ['BaseOCRPLModule']

import torch

from ocr.datasets.mix_dataset import MixDataset
from ocr.utils.builders import build_optimizer_from_cfg, build_lr_scheduler_from_cfg, \
    build_backbone_from_cfg, build_head_from_cfg, build_decoder_from_cfg, build_loss_from_cfg, build_transform_from_cfg, \
    build_dataset_from_cfg, build_metric_from_cfg, build_encoder_from_cfg
from ocr.utils.tokenizer import tokenize_vocab, TextCollate

CfgT = Dict[str, Any]


class BaseOCRPLModule(pl.LightningModule):
    def __init__(self,
                 vocab: List[str],
                 sequence_size: int,
                 backbone_cfg: CfgT = dict(),
                 encoder_cfg: Optional[CfgT] = None,
                 decoder_cfg: Optional[CfgT] = None,
                 head_cfg: Optional[CfgT] = None,
                 loss_cfgs: Optional[CfgT] = None,
                 metric_cfgs: List[CfgT] = list(),
                 train_transforms_cfg: CfgT = dict(),
                 val_transforms_cfg: CfgT = dict(),
                 train_dataset_cfg: CfgT = dict(),
                 val_dataset_cfg: CfgT = dict(),
                 train_dataloader_cfg: CfgT = dict(),
                 val_dataloader_cfg: CfgT = dict(),
                 optimizer_cfg: CfgT = dict(),
                 scheduler_cfg: Optional[CfgT] = None,
                 scheduler_update_params: Optional[CfgT] = None,
                 ):
        super(BaseOCRPLModule, self).__init__()
        self.backbone_cfg = backbone_cfg
        self.head_cfg = head_cfg
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.loss_cfgs = loss_cfgs
        self.metric_cfgs = metric_cfgs
        self.train_transforms_cfg = train_transforms_cfg
        self.val_transforms_cfg = val_transforms_cfg
        self.train_dataset_cfg = train_dataset_cfg
        self.val_dataset_cfg = val_dataset_cfg
        self.train_dataloader_cfg = train_dataloader_cfg
        self.val_dataloader_cfg = val_dataloader_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.scheduler_update_params = scheduler_update_params
        self.save_hyperparameters()

        self.scheduler = None  # Can be useful for LR logging in the progress bar
        self.sequence_size = sequence_size
        self.letter_to_token, self.token_to_letter = tokenize_vocab(vocab)
        self._build_models()

    def _build_models(self):
        self.backbone, in_ch = build_backbone_from_cfg(self.backbone_cfg.copy())
        self.head = None if self.head_cfg is None else build_head_from_cfg(in_ch, self.head_cfg)
        if self.head is not None:
            in_ch = self.head.output_channels
        self.encoder = None if self.encoder_cfg is None else build_encoder_from_cfg(in_ch, self.encoder_cfg.copy())
        self.decoder = None if self.decoder_cfg is None else build_decoder_from_cfg(self.decoder_cfg.copy())

        self.losses, self.metrics = [], torch.nn.ModuleList()
        for loss_cfg in self.loss_cfgs:
            loss_name = loss_cfg.pop('name') if 'name' in loss_cfg else loss_cfg['type'].lower()
            loss_module = build_loss_from_cfg(loss_cfg.copy())
            self.add_module(f'loss_{loss_name}', loss_module)
            self.losses.append((loss_name, loss_module))

        self._metric_names = []
        for metric_cfg in deepcopy(self.metric_cfgs):
            metric_name = metric_cfg.pop('name') if 'name' in metric_cfg else metric_cfg['type'].lower()
            if metric_cfg['type'] != 'SymbolRate':
                self._metric_names.append(metric_name)
                metric_module = build_metric_from_cfg(metric_cfg.copy())
                self.metrics.append(metric_module)
            else:
                vocab = metric_cfg.pop('vocab')
                name_prefix = metric_cfg.pop('name_prefix') if 'name_prefix' in metric_cfg else ''
                for v in vocab:
                    cfg = deepcopy(metric_cfg)
                    metric_name = f'{name_prefix}_{v}'
                    cfg['token'] = self.letter_to_token[v]
                    self._metric_names.append(metric_name)
                    metric_module = build_metric_from_cfg(cfg.copy())
                    self.metrics.append(metric_module)

    def forward_val(self, src, sequence_size):
        x = self.backbone(src)
        if self.head is not None:
            x = self.head(x)
        if self.encoder is not None:
            x = self.encoder(x)
        x = self.decoder(x, sequence_size)
        return x

    def forward(self, src):
        return self.forward_val(src, self.sequence_size)

    def train_forward(self, src, tgt):
        x = self.backbone(src)
        if self.head is not None:
            x = self.head(x)
        if self.encoder is not None:
            x = self.encoder(x)
        x = self.decoder.train_forward(x, tgt)
        return x

    def training_step(self, batch, batch_idx):
        images, tokens, lengths = batch
        output = self.train_forward(images, tokens[:-1, :])

        losses = []
        for loss_name, loss_module in self.losses:
            losses.append(loss_module(output, tokens.permute(1, 0)[:, 1:], lengths - 1))
            self.log(loss_name, losses[-1], prog_bar=True, on_epoch=False, on_step=True, logger=True)

        return torch.sum(torch.stack(losses))

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        images, tokens, lengths = batch
        output = self.forward_val(images, len(tokens))

        tokens = tokens.permute(1, 0)
        for metric_name, metric_module in zip(self._metric_names, self.metrics):
            metric_module.update(output, tokens)

    def validation_epoch_end(self, outputs):
        for metric_name, metric_module in zip(self._metric_names, self.metrics):
            if not metric_name.startswith('symbol_rate_'):
                self.log(f'{metric_name}', metric_module.compute(), prog_bar=True, on_epoch=True, logger=True)
            else:
                self.log(f'{metric_name}', metric_module.compute(), prog_bar=False, on_epoch=True, logger=True)
            metric_module.reset()

    def configure_optimizers(self):
        optimizer = build_optimizer_from_cfg(self.parameters(), self.optimizer_cfg.copy())
        if self.scheduler_cfg is not None:
            self.scheduler = build_lr_scheduler_from_cfg(optimizer, self.scheduler_cfg.copy())
            lr_scheduler_info = {
                'scheduler': self.scheduler,
                'name': 'lr',
                **self.scheduler_update_params}
            return [optimizer], [lr_scheduler_info]
        return [optimizer]

    def __create_dataloader(self, transforms_cfg, dataset_cfg, dataloader_cfg):
        if isinstance(dataset_cfg, list):
            datasets = []
            for cfg in dataset_cfg:
                dataset_name = cfg['name']
                transforms = build_transform_from_cfg(deepcopy(transforms_cfg[dataset_name]))
                datasets.append(build_dataset_from_cfg(transforms, deepcopy(cfg)))
            dataset = MixDataset(datasets)
        else:
            transforms = build_transform_from_cfg(transforms_cfg.copy())
            dataset = build_dataset_from_cfg(transforms, dataset_cfg.copy())
        return torch.utils.data.DataLoader(dataset, collate_fn=TextCollate(letter_to_token=self.letter_to_token),
                                           **dataloader_cfg)

    def train_dataloader(self):
        return self.__create_dataloader(self.train_transforms_cfg, self.train_dataset_cfg, self.train_dataloader_cfg)

    def val_dataloader(self):
        if isinstance(self.val_dataset_cfg, list):
            dataloaders = [self.__create_dataloader(self.val_transforms_cfg, dataset_cfg, self.val_dataloader_cfg) for
                           dataset_cfg in self.val_dataset_cfg]
        elif isinstance(self.val_dataset_cfg, dict):
            dataloaders = [
                self.__create_dataloader(self.val_transforms_cfg, self.val_dataset_cfg, self.val_dataloader_cfg)]
        else:
            assert False, 'incorrect val_dataset_cfg'
        return dataloaders
