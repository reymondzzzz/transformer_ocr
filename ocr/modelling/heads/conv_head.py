import torch

__all__ = ['ConvHead']


class ConvHead(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 conv_kernel_size: int = 1,
                 dropout_rate: float = 0.):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.output_channels = output_channels
        self.conv = torch.nn.Conv2d(input_channels, output_channels, conv_kernel_size, bias=False)
        self.bn = torch.nn.BatchNorm2d(output_channels)
        if self.dropout_rate != 0.:
            self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.dropout_rate != 0.:
            x = self.dropout(x)
        return x

    def _initialize_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        # self.conv.bias.data.zero_()

        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
