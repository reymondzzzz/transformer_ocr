import torch
from torch import nn

from ocr.blocks.attention import Attention
from ocr.utils.tokenizer import tokenize_vocab


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_features, embed_size, vocab, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_features
        self.embed_size = embed_size
        self.letter_to_token, _ = tokenize_vocab(vocab)
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attention('concat', hidden_features)
        self.gru = nn.GRU(hidden_features + embed_size, hidden_features, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_features, len(vocab))

    def forward_step(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1)  # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        # rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = self.out(output)
        return output, hidden

    def train_forward(self, encoder_outputs, target_seq):
        batch_size = encoder_outputs.size(0)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=encoder_outputs.device)

        output_logits = []
        for i in range(target_seq.size(0)):
            current_symbols = target_seq[i]
            output, decoder_hidden = self.forward_step(current_symbols, decoder_hidden, encoder_outputs)

            output_logits.append(output)

        output_logits = torch.cat(output_logits, dim=0)

        return output_logits.permute(1, 0, 2)

    def forward(self, encoder_outputs, max_seq_size):
        batch_size = encoder_outputs.size(0)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=encoder_outputs.device)

        def decode(step_output):
            symbols = step_output.topk(1)[1]
            return symbols

        sequences = torch.ones(1, batch_size, dtype=torch.long, device=encoder_outputs.device) * self.letter_to_token[
            'sos']
        for i in range(max_seq_size - 1):
            current_symbols = sequences[i]
            output, decoder_hidden = self.forward_step(current_symbols, decoder_hidden, encoder_outputs)
            last_symbols = decode(output)
            last_symbols = last_symbols.squeeze(2)
            sequences = torch.cat((sequences, last_symbols), dim=0)

        return sequences
