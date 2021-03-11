import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size: int = 3,
               padding: int = 0, stride: int = 1, dilation: int = 1, groups: int = 1,
               batch_norm: bool = False):
    seq = []
    seq.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, padding=padding, stride=stride,
                         groups=groups, dilation=dilation))
    if batch_norm:
        seq.append(nn.BatchNorm2d(num_features=out_channels))
    seq.append(nn.ReLU())
    return nn.Sequential(*seq)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc1 = conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=1, batch_norm=True)
        self.inc2 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=1, batch_norm=True),
            conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, batch_norm=True)
        )
        self.inc3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2, batch_norm=True)
        )

    def forward(self, x):
        x1 = self.inc1(x)
        x2 = self.inc2(x)
        x3 = self.inc3(x)
        return torch.cat((x1, x2, x3), dim=1)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = conv_block(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.layer2 = conv_block(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, batch_norm=True)
        self.layer3 = conv_block(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, batch_norm=True)

        self.layer4 = InceptionBlock(in_channels=32, out_channels=16)
        self.layer5 = InceptionBlock(in_channels=48, out_channels=16)
        self.layer6 = conv_block(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1, batch_norm=True)

        self.layer7 = InceptionBlock(in_channels=64, out_channels=32)
        self.layer8 = InceptionBlock(in_channels=96, out_channels=32)
        self.layer9 = conv_block(in_channels=96, out_channels=64, kernel_size=3, stride=2, padding=1, batch_norm=True)

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
