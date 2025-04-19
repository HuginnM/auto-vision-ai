import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double Convolutional UNet block. Contains two Conv2d layers
    followed by BatchNorm layers and RelU activation functions.

    :param in_channels: a number of input channels.
    :param out_channels: a number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the DoubleConv block.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, out_channels,  H, W].
        """
        x = self.conv(x)
        return x


class InConv(nn.Module):
    """
    An input block for UNet. A wrapper around DoubleConv block.

    :param in_channels: a number of input channels.
    :param out_channels: a number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        """
        Forward pass through the InConv block.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, out_channels,  H, W].
        """
        x = self.conv(x)
        return x


class Down(nn.Module):
    """
    UNet encoder block, which decreases spatial dimension by two and
    increases the number of channels.

    :param in_channels: a number of input channels.
    :param out_channels: a number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = nn.Sequential(nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        """
        Forward pass through the Down block.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, out_channels,  H/2, W/2].
        """
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upscaling UNet block. Takes the output of the previous layer and
    upscales it, increasing the spatial dim by a factor of 2.
    Then, takes the output of the corresponding down layer and concatenates it
    with the previous layer. Finally, passes them through the DoubleConv block.

    :param in_channels: a number of input channels.
    :param out_channels: a number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass through the Up block.

        :param x1: an output of the previous upscale layer with shape [batch, out_channels, H, W].
        :param x2: an output of the corresponding down layer with shape [batch, out_channels, 2*H, 2*W].
        :return: an output with shape [batch, out_channels,  2*H, 2*W].
        """
        x = self.up(x1)
        x = torch.concat([x2, x], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """
    UNet output layer, which decreases the number of channels.

    :param in_channels: a number of input channels.
    :param n_classes: a number classes.
    """

    def __init__(self, in_channels, n_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the OutConv block.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, n_classes,  H, W].
        """
        x = self.conv(x)
        return x


class Unet(nn.Module):
    """
    The complete architecture of UNet using layers defined above.

    :param in_channels: a number of input channels.
    :param n_classes: a number classes.
    """

    def __init__(self, in_channels, n_classes):
        super(Unet, self).__init__()
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512 * 2, 256)
        self.up2 = Up(256 * 2, 128)
        self.up3 = Up(128 * 2, 64)
        self.up4 = Up(64 * 2, 64)
        self.out = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the Neural Network.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, n_classes,  H, W].
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out(x)
        return out
