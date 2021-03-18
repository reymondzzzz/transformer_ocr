from torch import nn


class AttentionEncoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_channels, feature_x, feature_y, hidden_features):
        super(AttentionEncoder, self).__init__()
        self._input_channels = input_channels
        self._feature_x = feature_x
        self._feature_y = feature_y
        self.encode_emb = nn.Linear(input_channels, hidden_features)

    def forward(self, input):
        b = input.size(0)
        input = input.reshape(b, self._input_channels, -1).permute(2, 0, 1)
        encoder_outputs = self.encode_emb(input)
        return encoder_outputs
