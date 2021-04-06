import random

import torch
import torch.nn.functional as F
from torch import nn

from ocr.utils.tokenizer import tokenize_vocab


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_features, vocab, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_features
        self.output_size = len(vocab)
        self.letter_to_token, _ = tokenize_vocab(vocab)
        self.dropout_p = dropout_p

        self.embedding = nn.Linear(self.output_size, self.hidden_size, bias=False)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.vat = nn.Linear(hidden_features, 1)

    # torch one_hot is not converted in trt
    @staticmethod
    def _to_one_hot(y, num_classes):
        return torch.eye(num_classes)[y]

    def emb(self, symbols):
        one_hot = self._to_one_hot(symbols, num_classes=self.output_size)
        out = self.embedding(one_hot.float())
        return out

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.emb(input)
        embedded = self.dropout(embedded)

        batch_size = encoder_outputs.shape[1]
        alpha = hidden + encoder_outputs
        alpha = alpha.contiguous().view(-1, alpha.shape[-1])
        attn_weights = self.vat(torch.tanh(alpha))
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute((1, 0, 2)))
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def _decode_logits(self, logits):
        symbols = logits.topk(1)[1]
        return symbols

    def train_forward(self, encoder_outputs, target_seq):
        batch_size = encoder_outputs.size(1)
        hidden = self.init_hidden(batch_size).to(encoder_outputs.device)

        pre_output_vectors = []
        teach_forcing = random.random() > 0.5
        current_symbols = target_seq[0]
        if teach_forcing:
            for i in range(1, target_seq.size(0)):
                output, hidden, attn_weights = self.forward_step(current_symbols, hidden, encoder_outputs)
                pre_output_vectors.append(output.unsqueeze(1))
                current_symbols = target_seq[i]
        else:
            for i in range(1, target_seq.size(0)):
                output, hidden, attn_weights = self.forward_step(current_symbols, hidden, encoder_outputs)
                current_symbols = self._decode_logits(output).squeeze(1).detach()
                pre_output_vectors.append(output.unsqueeze(1))

        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return pre_output_vectors

    def forward(self, encoder_outputs, max_seq_size):
        batch_size = encoder_outputs.size(1)
        hidden = self.init_hidden(batch_size).to(encoder_outputs.device)

        sequences = torch.ones(batch_size, 1, dtype=torch.long, device=encoder_outputs.device) * self.letter_to_token[
            'sos']

        coords = []
        for i in range(max_seq_size - 1):
            current_symbols = sequences[:, i]
            output, hidden, attn_weights = self.forward_step(current_symbols, hidden, encoder_outputs)
            coords.append(attn_weights.view(batch_size, -1).max(1)[1].unsqueeze(1))
            last_symbols = self._decode_logits(output)
            sequences = torch.cat((sequences, last_symbols), dim=1)

        # texts, coordinates of symbol centers by feature map
        return sequences, torch.cat(coords, dim=1)

    def init_hidden(self, batch_size):
        result = torch.zeros(1, batch_size, self.hidden_size)

        return result
