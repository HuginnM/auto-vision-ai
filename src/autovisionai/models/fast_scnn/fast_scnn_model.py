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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, stride=stride, bias=False),
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


class DSConv(nn.Module):
    """
    Depthwise Separable Convolutions.

    :param dw_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    :param stride: a stride of the conv.
    """
    def __init__(self, dw_channels, out_channels, stride=1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, kernel_size=[3, 3], padding=1, 
                      stride=stride, bias=False, groups=dw_channels),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(dw_channels, out_channels, kernel_size=[1, 1], padding=0, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the DSConv block.

        :param x: an input with dw_channels.
        :return: an output with out_channels.
        """
        out = self.conv(x)
        return out


class DWConv(nn.Module):
    """
    Depthwise Convolutions.

    :param dw_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    :param stride: a stride of the conv.
    """
    def __init__(self, dw_channels, out_channels, stride=1):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, kernel_size=[3, 3], padding=1, 
                      stride=stride, bias=False, groups=dw_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the DWConv block.

        :param x: an input with dw_channels.
        :return: an output with out_channels.
        """
        out = self.conv(x)
        return out


class LinearBottleneck(nn.Module):
    """
    Linear Bottleneck.
    Allows to enable high accuracy and performance.

    :param in_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    :param t: an expansion factor.
    :param stride: a stride of the conv.
    """
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super(LinearBottleneck, self).__init__()
        # use residual add if features match, otherwise a normal Sequential
        self.use_shortcut = True if in_channels == out_channels and stride == 1 else False

        # nn.Sequential should contain ConvBNReLU(in_channels, in_channels * t, kernel_size=1) -> 
        # DWConv(in_channels * t, in_channels * t, stride) -> 
        # Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False) -> BatchNorm2d.
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, in_channels*t, kernel_size=1),
            DWConv(in_channels*t, in_channels*t, stride=stride),
            nn.Conv2d(in_channels*t, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """
        Forward pass through the LinearBottleneck block.

        :param x: an input with in_channels.
        :return: an output with out_channels.
        """
        # Use conv block
        out = self.block(x)
        # Check use_shortcut or not
        if self.use_shortcut:
            out = x + out
        return out


