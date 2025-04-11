from autovisionai.models.fast_scnn.fast_scnn_model import *


def test_conv_bn_relu():
    test_input = torch.randn((2, 3, 256, 256))

    conv = ConvBNReLU(3, 64)
    output = conv(test_input)
    assert output.shape == torch.Size([2, 64, 254, 254])

    params_conv_bn_relu = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    assert params_conv_bn_relu == 1856


def test_ds_conv():
    test_input = torch.randn((2, 3, 256, 256))

    conv = DSConv(3, 32)
    output = conv(test_input)
    assert output.shape == torch.Size([2, 32, 256, 256])

    params_depthwise_sep = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    assert params_depthwise_sep == 193


def test_dw_conv():
    test_input = torch.randn((2, 3, 256, 256))

    conv = DWConv(3, 27)
    output = conv(test_input)
    assert output.shape == torch.Size([2, 27, 256, 256])

    params_depthwise = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    assert params_depthwise == 297


def test_linear_bottleneck():
    test_input = torch.randn((2, 3, 256, 256))

    conv = LinearBottleneck(3, 64)
    output = conv(test_input)
    assert output.shape == torch.Size([2, 64, 128, 128])

    params_linear_bottleneck = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    assert params_linear_bottleneck == 1568


def test_pyramid_pooling():
    pp_module = PyramidPooling(128, 128)

    test_input_to_upsample = torch.randn((2, 3, 256, 256))
    upsampled_output = pp_module.upsample(test_input_to_upsample, (8, 8))
    assert upsampled_output.shape == torch.Size([2, 3, 8, 8])

    test_input_to_pool_and_forward = torch.randn((2, 128, 8, 8))
    pooled_output = pp_module.pool(test_input_to_pool_and_forward, 2)
    assert pooled_output.shape == torch.Size([2, 128, 2, 2])

    forward_output = pp_module(test_input_to_pool_and_forward)
    assert forward_output.shape == torch.Size([2, 128, 8, 8])

    params_pp_module = sum(p.numel() for p in pp_module.parameters() if p.requires_grad)
    assert params_pp_module == 49664
