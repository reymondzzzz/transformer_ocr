import torch
from torch import nn


class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot, self).__init__()
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        emb.weight.requires_grad = False
        self.emb = emb

    def forward(self, input_):
        return self.emb(input_)


class AttentionEncoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_channels, feature_x, feature_y, hidden_features):
        super(AttentionEncoder, self).__init__()
        self._input_channels = input_channels
        self._feature_x = feature_x
        self._feature_y = feature_y
        self.onehot_x = OneHot(feature_x)
        self.onehot_y = OneHot(feature_y)
        self.encode_emb = nn.Linear(input_channels + feature_x + feature_y, hidden_features)

    def forward(self, input):
        b, fc, fh, fw = input.size()
        x, y = torch.meshgrid(torch.arange(fh, device=input.device), torch.arange(fw, device=input.device))

        h_loc = self.onehot_x(x)
        w_loc = self.onehot_y(y)

        loc = torch.cat([h_loc, w_loc], dim=2).unsqueeze(0).expand(b, -1, -1, -1)

        encoder_outputs = torch.cat([input.permute(0, 2, 3, 1), loc], dim=3)
        encoder_outputs = encoder_outputs.contiguous().view(b, -1,
                                                            self._input_channels + self._feature_x + self._feature_y)
        encoder_outputs = self.encode_emb(encoder_outputs)
        return encoder_outputs
