from torch import nn


class LinearHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(LinearHead, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self._initialize_weights()

    def forward(self, x):
        return self.fc(x)

    def _initialize_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
