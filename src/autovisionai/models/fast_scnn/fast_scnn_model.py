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
        self.use_shortcut = True if in_channels == out_channels and stride == 1 else False

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
        out = self.block(x)

        if self.use_shortcut:
            out = x + out

        return out


class PyramidPooling(nn.Module):
    """
    Pyramid pooling module.
    Aggregates the different-region-based context information.

    :param in_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    """

    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        inter_channels = in_channels // 4
        self.conv1 = ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv2 = ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv3 = ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv4 = ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.out = ConvBNReLU(in_channels*2, out_channels, kernel_size=1)

    @staticmethod
    def upsample(x, size):
        """
        Up samples the input.

        :param x: an input.
        :param size: a size to up sample the input.
        :return: an up sampled input.
        """
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    @staticmethod
    def pool(x, size):
        """
        Applies a 2D adaptive average pooling over an input.

        :param x: an input.
        :param size: the target output size.
        :return: a pooled input.
        """
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        """
        Forward pass through the PyramidPooling block.

        :param x: an input.
        :return: an output of the PPM.
        """
        size = x.shape[-2:]
        feat1 = PyramidPooling.upsample(self.conv1(PyramidPooling.pool(x, size=1)), size)
        feat2 = PyramidPooling.upsample(self.conv2(PyramidPooling.pool(x, size=2)), size)
        feat3 = PyramidPooling.upsample(self.conv3(PyramidPooling.pool(x, size=3)), size)
        feat4 = PyramidPooling.upsample(self.conv4(PyramidPooling.pool(x, size=6)), size)

        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """
    Learning to downsample module.

    :param dw_channels1: a number of channels to downsample.
    :param dw_channels2: a number of channels to downsample.
    :param out_channels: a number of channels produced by the conv.
    """

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        """
        Forward pass through the LearningToDownsample block.

        :param x: an input.
        :return: an output of the LTD.
        """
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """
    Global feature extractor module.

    :param in_channels: number of channels in the input.
    :param block_channels: list with number of channels produced by the LinearBottleneck.
    :param out_channels: number of output channels.
    :param t: an expansion factor.
    :param num_blocks: number of times block is repeated.
    """

    def __init__(
        self,
        in_channels=64,
        block_channels=(64, 96, 128),
        out_channels=128,
        t=6,
        num_blocks=(3, 3, 3),
    ):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = GlobalFeatureExtractor._make_layer(
            block=LinearBottleneck,
            inplanes=in_channels,
            planes=block_channels[0],
            blocks=num_blocks[0],
            t=t,
            stride=2,
        )
        self.bottleneck2 = GlobalFeatureExtractor._make_layer(
            block=LinearBottleneck,
            inplanes=block_channels[0],
            planes=block_channels[1],
            blocks=num_blocks[1],
            t=t,
            stride=2,
        )
        self.bottleneck3 = GlobalFeatureExtractor._make_layer(
            block=LinearBottleneck,
            inplanes=block_channels[1],
            planes=block_channels[2],
            blocks=num_blocks[2],
            t=t,
            stride=1,
        )
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    @staticmethod
    def _make_layer(block, inplanes, planes, blocks, t=6, stride=1):
        """

        :param block: block to create.
        :param inplanes: number of input channels.
        :param planes: number of output channels.
        :param blocks: number of times block is repeated.
        :param t: an expansion factor.
        :param stride: a stride of the conv.
        :return: nn.Sequential of layers.
        """
        layers = [block(inplanes, planes, t, stride)]
        for _i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the GlobalFeatureExtractor block.

        :param x: an input.
        :return: an output of the GFE.
        """
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """
    Feature fusion module.

    :param highter_in_channels: high resolution channels input.
    :param lower_in_channels:  low resolution channels input.
    :param out_channels: number of output channels.
    :param scale_factor: scale factor.
    """
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DWConv(lower_in_channels, lower_in_channels)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, higher_res_feature, lower_res_feature):
        """
        Forward pass through the FeatureFusionModule block.

        :param higher_res_feature: high resolution features.
        :param lower_res_feature: low resolution features.
        :return: an output of the FFM.
        """
        lower_res_feature = F.interpolate(
            lower_res_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = self.relu(lower_res_feature + higher_res_feature)
        return out


class Classifier(nn.Module):
    """
    Classifier.

    :param dw_channels: number of channels for dsconv.
    :param num_classes: number of classes.
    :param stride: a stride of the conv.
    """
    def __init__(self, dw_channels, num_classes, stride=1):
        super(Classifier, self).__init__()
        self.dsconv1 = DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(dw_channels, num_classes, kernel_size=1))

    def forward(self, x):
        """
        Forward pass through the Classifier block.

        :param x: an input.
        :return: an output of the Classifier.
        """
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class FastSCNN(nn.Module):
    """
    The complete architecture of FastSCNN using layers defined above.

    :param num_classes: number of classes.
    """
    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64,
            block_channels=[64, 96, 128],
            out_channels=128,
            t=6,
            num_blocks=[3, 3, 3]
        )
        self.feature_fusion = FeatureFusionModule(
            highter_in_channels=64, lower_in_channels=128, out_channels=128)
        self.classifier = Classifier(dw_channels=128, num_classes=num_classes)

    def forward(self, x):
        """
        Forward pass through the FastSCNN.

        :param x: an input.
        :return: an output of the FastSCNN.
        """
        size = x.shape[-2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        out = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return out
