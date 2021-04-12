import torch

from .base_ocr_module import BaseOCRPLModule


class AttentionOCRPLModule(BaseOCRPLModule):
    def training_step(self, batch, batch_idx):
        images, tokens, lengths = batch
        output = self.train_forward(images, tokens)

        losses = []
        for loss_name, loss_module in self.losses:
            losses.append(loss_module(output, tokens.permute(1, 0)[:, 1:], lengths - 1))
            self.log(loss_name, losses[-1], prog_bar=True, on_epoch=False, on_step=True, logger=True)

        return torch.sum(torch.stack(losses))

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        images, tokens, lengths = batch
        output, _, _ = self.forward_val(images, len(tokens))

        tokens = tokens.permute(1, 0)
        for metric_name, metric_module in zip(self._metric_names, self.metrics):
            metric_module.update(output, tokens)
