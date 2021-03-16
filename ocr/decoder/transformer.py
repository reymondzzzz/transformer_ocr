from typing import List

import torch
from torch import nn

from ocr.blocks import PositionalEncoding
from ocr.utils.tokenizer import tokenize_vocab


class TransformerDecoder(nn.Module):
    def __init__(self, vocab: List[str], hidden_features: int,
                 enc_layers=1, dec_layers=1, nhead=1, dropout=0.1):
        super().__init__()

        self.letter_to_token, _ = tokenize_vocab(vocab)

        self.pos_encoder = PositionalEncoding(hidden_features, dropout)
        self.decoder = nn.Embedding(len(vocab), hidden_features)
        self.pos_decoder = PositionalEncoding(hidden_features, dropout)
        self.transformer = nn.Transformer(d_model=hidden_features, nhead=nhead, num_encoder_layers=enc_layers,
                                          num_decoder_layers=dec_layers, dim_feedforward=hidden_features * 4,
                                          dropout=dropout,
                                          activation='relu')

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
        self.fc = nn.Linear(hidden_features, len(vocab), bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        self.fc.weight.data.normal_(0, 0.01)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def make_len_mask(inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, seq_size):
        batch_size = src.shape[0]

        x = src.flatten(2).permute(2, 0, 1)
        memory = self.transformer.encoder(self.pos_encoder(x))

        trg_tensor = torch.ones((1, batch_size), dtype=torch.long, device=src.device)
        for i in range(seq_size - 1):
            output = self.decoder(trg_tensor)
            output = self.pos_decoder(output)
            output = self.transformer.decoder(output, memory)
            output = self.fc(output)
            out_token = output[i].argmax(1).unsqueeze(0)
            trg_tensor = torch.cat((trg_tensor, out_token), dim=0)
        return trg_tensor

    def train_forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        x = src.flatten(2).permute(2, 0, 1)
        src_pad_mask = self.make_len_mask(x[:, :, 0])
        src = self.pos_encoder(x)

        trg_pad_mask = self.make_len_mask(trg)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
                                  memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
                                  memory_key_padding_mask=src_pad_mask)
        output = self.fc(output)
        return output.permute(1, 0, 2)
