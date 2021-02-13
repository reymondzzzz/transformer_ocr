import math
from typing import List

import torch
from torch import nn

from ocr.utils.tokenizer import tokenize_vocab

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


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
        #bs ch h w
        x = src.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
        memory = self.transformer.encoder(self.pos_encoder(x))

        out_tokens = torch.LongTensor([[self.letter_to_token['sos'], ]] * batch_size).permute(1, 0)
        for i in range(seq_size - 1):
            trg_tensor = torch.LongTensor(out_tokens).to(src.device)

            output = self.decoder(trg_tensor)
            output = self.pos_decoder(output)
            output = self.transformer.decoder(output, memory)
            output = self.fc(output)
            out_token = output[-1, None].argmax(2).detach().cpu()
            out_tokens = torch.cat((out_tokens, out_token), dim=0)
            # out_indexes.append(out_token)
        return out_tokens.to(src.device)

    # def forward(self, src, l):
    #     x = src.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
    #
    #     memory = self.transformer.encoder(self.pos_encoder(x))
    #
    #     out_indexes = [self.letter_to_token['sos'], ]
    #
    #     for i in range(l - 1):
    #         trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(src.device)
    #
    #         output = self.fc(self.transformer.decoder(self.pos_decoder(self.decoder(trg_tensor)), memory))
    #         out_token = output.argmax(2)[-1].item()
    #         out_indexes.append(out_token)
    #     return torch.tensor(out_indexes).unsqueeze(1).to(src.device)

    def train_forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        x = src.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
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
        return output
