import torch


def tokenize_vocab(vocab):
    letter_to_token, token_to_letter = {}, {}
    for idx, letter in enumerate(vocab):
        letter_to_token[letter] = idx
        token_to_letter[idx] = letter
    return letter_to_token, token_to_letter


class TextCollate:
    def __init__(self, letter_to_token):
        self.letter_to_token = letter_to_token

    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])
        lengths = torch.LongTensor([i[1].size(0) for i in batch])
        y_padded = torch.full((max_y_len, len(batch)), self.letter_to_token['pad'], dtype=torch.long)

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        return x_padded, y_padded, lengths
