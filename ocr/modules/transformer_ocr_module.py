from copy import deepcopy
from typing import Dict, Any, Optional, List

import pytorch_lightning as pl

__all__ = ['TransformerOCRPLModule']

import torch

from ocr.utils.builders import build_optimizer_from_cfg, build_lr_scheduler_from_cfg, \
    build_backbone_from_cfg, build_head_from_cfg, build_decoder_from_cfg, build_decoder_head_from_cfg, \
    build_loss_from_cfg, build_transform_from_cfg, build_dataset_from_cfg, build_metric_from_cfg
from ocr.utils.tokenizer import tokenize_vocab

CfgT = Dict[str, Any]

class TextCollate():
    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        return x_padded, y_padded

class TransformerOCRPLModule(pl.LightningModule):
    def __init__(self,
                 vocab: List[str],
                 sequence_size: int,
                 backbone_cfg: CfgT = dict(),
                 decoder_cfg: CfgT = dict(),
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
                 scheduler_cfg: CfgT = dict(),
                 scheduler_update_params: CfgT = dict(),
                 ):
        super(TransformerOCRPLModule, self).__init__()
        self.backbone_cfg = backbone_cfg
        self.head_cfg = head_cfg
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
        self._build_models()
        self.sequence_size = sequence_size
        self.letter_to_token, self.token_to_letter = tokenize_vocab(vocab)

    def _build_models(self):
        self.backbone, in_ch = build_backbone_from_cfg(self.backbone_cfg.copy())
        self.head = None if self.head_cfg is None else build_head_from_cfg(in_ch, self.head_cfg)
        self.decoder = build_decoder_from_cfg(self.decoder_cfg.copy())

        self.losses, self.metrics = [], torch.nn.ModuleList()
        for loss_cfg in self.loss_cfgs:
            loss_name = loss_cfg.pop('name') if 'name' in loss_cfg else loss_cfg['type'].lower()
            loss_module = build_loss_from_cfg(loss_cfg.copy())
            self.add_module(f'loss_{loss_name}', loss_module)
            self.losses.append((loss_name, loss_module))

        self._metric_names = []
        for metric_cfg in deepcopy(self.metric_cfgs):
            metric_name = metric_cfg.pop('name') if 'name' in metric_cfg else metric_cfg['type'].lower()
            self._metric_names.append(metric_name)
            metric_module = build_metric_from_cfg(metric_cfg.copy())
            self.metrics.append(metric_module)

    def forward(self, src, sequence_size):
        x = self.backbone(src)
        if self.head is not None:
            x = self.head(x)
        x = self.decoder(x, sequence_size)
        return x

    def train_forward(self, src, tgt):
        x = self.backbone(src)
        if self.head is not None:
            x = self.head(x)
        x = self.decoder.train_forward(x, tgt)
        return x

    def training_step(self, batch, batch_idx):
        images, tokens = batch
        output = self.train_forward(images, tokens[:-1, :])

        losses = []
        for loss_name, loss_module in self.losses:
            losses.append(loss_module(output.view(-1, output.shape[-1]), torch.reshape(tokens[1:, :], (-1,))))
            self.log(loss_name, losses[-1], prog_bar=True, on_epoch=False, on_step=True, logger=True)

        self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, on_step=True, logger=False)
        return torch.sum(torch.stack(losses))

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        images, tokens = batch
        output = self(images, len(tokens))

        for metric in self.metrics:
            metric(output, tokens)

    def validation_epoch_end(self, outputs):
        for metric_name, metric_module in zip(self._metric_names, self.metrics):
            self.log(f'{metric_name}', metric_module.compute(), prog_bar=True, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = build_optimizer_from_cfg(self.parameters(), self.optimizer_cfg.copy())
        self.scheduler = build_lr_scheduler_from_cfg(optimizer, self.scheduler_cfg.copy())
        lr_scheduler_info = {'scheduler': self.scheduler, **self.scheduler_update_params}
        return [optimizer], [lr_scheduler_info]

    @staticmethod
    def __create_dataloader(transforms_cfg, dataset_cfg, dataloader_cfg):
        transforms = build_transform_from_cfg(transforms_cfg.copy())
        dataset = build_dataset_from_cfg(transforms, dataset_cfg.copy())
        return torch.utils.data.DataLoader(dataset, collate_fn=TextCollate(), **dataloader_cfg)

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

        # self._val_dataset_names = [dl.dataset.name for dl in dataloaders]
        # for ds_name in self._val_dataset_names:
        #     for metric_cfg in deepcopy(self.metric_cfgs):
        #         metric_name = metric_cfg.pop('name') if 'name' in metric_cfg else metric_cfg['type'].lower()
        #         self._metric_names.add(metric_name)
        #         metric_module = build_metric_from_cfg(metric_cfg.copy())
        #         self.metrics.append(metric_module)
        #         # self.metrics.append((ds_name, metric_name, metric_module))
        return dataloaders
