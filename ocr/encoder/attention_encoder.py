import torch
from torch import nn


class AttentionEncoder(nn.Module):
    def __init__(self, input_channels, feature_x, feature_y, hidden_features):
        super(AttentionEncoder, self).__init__()
        self._input_channels = input_channels
        self._feature_x = feature_x
        self._feature_y = feature_y
        self.encode_emb = nn.Linear(self._input_channels + self._feature_x + self._feature_y, hidden_features,
                                    bias=False)

    def _encode_coordinates(self, input):
        bs = input.size(0)
        h, w = input.shape[2:4]
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h))
        w_loc = torch.nn.functional.one_hot(x, num_classes=w)
        h_loc = torch.nn.functional.one_hot(y, num_classes=h)
        loc = torch.cat([h_loc, w_loc], 2)
        loc = loc.unsqueeze(0).repeat(bs, 1, 1, 1)
        return torch.cat([input, loc.permute(0, 3, 1, 2).to(input.device)], dim=1)

    def forward(self, input):
        b, c, h, w = input.size()
        input = self._encode_coordinates(input)
        input = input.reshape(b, self._input_channels + self._feature_x + self._feature_y, -1)
        input = input.permute(0, 2, 1).contiguous()
        input = self.encode_emb(input)

        input = input.permute(1, 0, 2)
        return input
