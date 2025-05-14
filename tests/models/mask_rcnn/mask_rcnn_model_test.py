import torch
from numpy.testing import assert_almost_equal

from autovisionai.core.models.mask_rcnn.mask_rcnn_model import *


def set_seed(seed):
    torch.manual_seed(seed)


def test_create_model():
    set_seed(42)
    test_input = torch.randn((1, 3, 128, 256))

    mask_rcnn = create_model(2)

    assert (
        str(mask_rcnn.roi_heads.mask_predictor).replace(" ", "")
        == """MaskRCNNPredictor(
                    (conv5_mask): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
                    (relu): ReLU(inplace=True)
                    (mask_fcn_logits): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
                    )""".replace(" ", "")
    )

    assert (
        str(mask_rcnn.roi_heads.box_predictor).replace(" ", "")
        == """FastRCNNPredictor(
                            (cls_score): Linear(in_features=1024, out_features=2, bias=True)
                            (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
                            )""".replace(" ", "")
    )

    mask_rcnn.eval()
    output = mask_rcnn.forward(test_input)
    assert_almost_equal(output[0]["boxes"].mean().item(), 101.84027099609375, decimal=5)

    params_mask_rcnn = sum(p.numel() for p in mask_rcnn.parameters() if p.requires_grad)
    assert params_mask_rcnn == 43699995
