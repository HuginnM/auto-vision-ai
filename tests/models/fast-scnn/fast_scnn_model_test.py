from autovisionai.core.models.fast_scnn.fast_scnn_model import *


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


def test_learning_to_downsample():
    test_input = torch.randn((2, 3, 256, 256))

    ldm_module = LearningToDownsample()

    output = ldm_module(test_input)
    assert output.shape == torch.Size([2, 64, 8, 8])

    params_ldm_module = sum(p.numel() for p in ldm_module.parameters() if p.requires_grad)
    assert params_ldm_module == 6640


def test_global_feature_extractor():
    test_input = torch.randn((2, 64, 32, 32))

    gfe_module = GlobalFeatureExtractor()
    output = gfe_module(test_input)
    assert output.shape == torch.Size([2, 128, 8, 8])

    params_gfe_module = sum(p.numel() for p in gfe_module.parameters() if p.requires_grad)
    assert params_gfe_module == 1066112


def test_feature_fusion_module():
    test_input = torch.randn((2, 128, 8, 8))
    test_higher_res_features = torch.randn((2, 64, 32, 32))

    ffm_module = FeatureFusionModule(64, 128, 128)
    output = ffm_module(test_higher_res_features, test_input)
    assert output.shape == torch.Size([2, 128, 32, 32])

    params_ffm_module = sum(p.numel() for p in ffm_module.parameters() if p.requires_grad)
    assert params_ffm_module == 26752


def test_classifier():
    test_input = torch.randn((2, 128, 32, 32))

    classifier = Classifier(128, 1)
    output = classifier(test_input)
    assert output.shape == torch.Size([2, 1, 32, 32])

    params_classifier = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    assert params_classifier == 36225


def test_full_fast_scnn():
    test_input = torch.randn((2, 3, 256, 256))

    fast_rcnn = FastSCNN(1)
    output = fast_rcnn(test_input)
    assert output.shape == torch.Size([2, 1, 256, 256])

    params_fast_rcnn = sum(p.numel() for p in fast_rcnn.parameters() if p.requires_grad)
    assert params_fast_rcnn == 1135729
