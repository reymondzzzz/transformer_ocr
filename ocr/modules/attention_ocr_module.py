import torch

from .transformer_ocr_module import TransformerOCRPLModule


class AttentionOCRPLModule(TransformerOCRPLModule):
    def training_step(self, batch, batch_idx):
        images, tokens, lengths = batch
        output = self.train_forward(images, tokens)

        losses = []
        for loss_name, loss_module in self.losses:
            losses.append(loss_module(output, tokens.permute(1, 0), lengths - 1))
            self.log(loss_name, losses[-1], prog_bar=True, on_epoch=False, on_step=True, logger=True)

        return torch.sum(torch.stack(losses))
