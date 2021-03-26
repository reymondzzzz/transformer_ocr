import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3,
                 padding: int = 0, dilation: int = 1, groups: int = 1,
                 batch_norm: bool = False, pooling: bool = False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding,
                              groups=groups, dilation=dilation, bias=False)
        self.activation = nn.ReLU()
        self.bn, self.max_pool = None, None
        if batch_norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        if pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, batch_norm=True)
        self.inc2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, batch_norm=True),
            ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, batch_norm=True)
        )
        self.inc3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2, batch_norm=True)
        )

    def forward(self, x):
        x1 = self.inc1(x)
        x2 = self.inc2(x)
        x3 = self.inc3(x)
        return torch.cat((x1, x2, x3), dim=1)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = ConvBlock(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.layer2 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, padding=1, batch_norm=True,
                                pooling=True)
        self.layer3 = ConvBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1, batch_norm=True)

        self.layer4 = InceptionBlock(in_channels=32, out_channels=16)
        self.layer5 = InceptionBlock(in_channels=48, out_channels=16)
        self.layer6 = ConvBlock(in_channels=48, out_channels=64, kernel_size=3, padding=1, batch_norm=True,
                                pooling=True)

        self.layer7 = InceptionBlock(in_channels=64, out_channels=32)
        self.layer8 = InceptionBlock(in_channels=96, out_channels=32)
        self.layer9 = ConvBlock(in_channels=96, out_channels=64, kernel_size=3, padding=1, batch_norm=True)

    @property
    def output_channels(self):
        return 64

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        return x
