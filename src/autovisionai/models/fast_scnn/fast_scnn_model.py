import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """
    Conv-BatchNorm-ReLU block.

    :param in_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    :param kernel_size: a size of the conv kernel.
    :param stride: a stride of the conv.
    :param padding: padding added to all four sides of the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the ConvBNReLU block.

        :param x: an input with in_channels.
        :return: an output with out_channels.
        """
        out = self.conv(x)
        return out
