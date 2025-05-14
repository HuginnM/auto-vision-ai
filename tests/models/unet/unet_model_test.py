from numpy.testing import assert_almost_equal

from autovisionai.core.models.unet.unet_model import *


def set_seed(seed):
    torch.manual_seed(seed)


def test_doubleconv():
    test_input = torch.randn((1, 3, 512, 512))

    conv = DoubleConv(3, 6)
    output = conv(test_input)
    assert output.shape == torch.Size([1, 6, 512, 512])

    params_doubleconv = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    assert params_doubleconv == 522


def test_down():
    test_input = torch.randn((1, 3, 512, 512))

    down_conv = Down(3, 6)
    output = down_conv(test_input)
    assert output.shape == torch.Size([1, 6, 256, 256])

    params_down_conv = sum(p.numel() for p in down_conv.parameters() if p.requires_grad)
    assert params_down_conv == 522


def test_up():
    x1 = torch.randn((1, 8, 32, 32))
    x2 = torch.randn((1, 8, 64, 64))

    up_conv = Up(16, 8)
    output = up_conv(x1, x2)
    assert output.shape == torch.Size([1, 8, 64, 64])

    params_up_conv = sum(p.numel() for p in up_conv.parameters() if p.requires_grad)
    assert params_up_conv == 2040


def test_outblock():
    set_seed(42)
    test_input = torch.randn((1, 8, 16, 16))

    out_conv = OutConv(8, 8)
    output = out_conv(test_input)
    assert_almost_equal(output.mean().item(), 0.15203364193439484, decimal=5)

    params_out_conv = sum(p.numel() for p in out_conv.parameters() if p.requires_grad)
    assert params_out_conv == 72


def test_full_unet():
    set_seed(42)
    test_input = torch.randn((4, 3, 64, 64))

    unet = Unet(3, 1)
    output = unet(test_input)
    assert output.shape == torch.Size([4, 1, 64, 64])
    assert_almost_equal(output.mean().item(), 0.011525428853929043, decimal=5)

    params_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    assert params_unet == 14788929
