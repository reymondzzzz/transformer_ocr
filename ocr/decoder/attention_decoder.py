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

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.v_at = nn.Linear(hidden_features, 1)

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        batch_size = encoder_outputs.shape[1]
        alpha = hidden + encoder_outputs
        alpha = alpha.view(-1, alpha.shape[-1])
        attn_weights = self.v_at(torch.tanh(alpha))
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.matmul(attn_weights,
                                    encoder_outputs.permute((1, 0, 2)))
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def init_hidden(self, encoder_outputs):
        return torch.zeros(1, encoder_outputs.size(1), self.hidden_size, device=encoder_outputs.device)

    def train_forward(self, encoder_outputs, target_seq):
        hidden = self.init_hidden(encoder_outputs)

        target_seq = target_seq.permute(1, 0)

        pre_output_vectors = []
        for i in range(target_seq.size(1)):
            current_symbols = target_seq[:, i]
            output, hidden, attn_weights = self.forward_step(current_symbols, hidden, encoder_outputs)
            pre_output_vectors.append(output.unsqueeze(1))

        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return pre_output_vectors

    def forward(self, encoder_outputs, max_seq_size):
        batch_size = encoder_outputs.size(1)
        hidden = self.init_hidden(encoder_outputs)

        def decode(step_output):
            symbols = step_output.topk(1)[1]
            return symbols

        sequences = torch.ones(batch_size, 1, dtype=torch.long, device=encoder_outputs.device) * self.letter_to_token[
            'sos']

        for i in range(max_seq_size - 1):
            current_symbols = sequences[:, i]
            output, hidden, attn_weights = self.forward_step(current_symbols, hidden, encoder_outputs)
            output = F.log_softmax(output, dim=1)
            last_symbols = decode(output)
            sequences = torch.cat((sequences, last_symbols), dim=1)

        return sequences.permute(1, 0)
