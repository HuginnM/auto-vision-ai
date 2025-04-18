from typing import Tuple

import numpy as np
import torch

from autovisionai.configs.config import CONFIG
from autovisionai.models.mask_rcnn.mask_rcnn_model import create_model


def model_inference(trained_model_path: str, image: torch.Tensor) -> Tuple[np.ndarray, ...]:
    """
    Loads the entire trained Mask R CNN model and predicts boxes, labels, scores and
    segmentation masks for an input image.

    :param trained_model_path: a path to saved model.
    :param image: a torch tensor with a shape [1, 3, H, W].
    :return: predicted numpy boxes, labels, scores, masks for an input image.
    """

    model = create_model(n_classes=CONFIG['mask_rcnn']['model']['n_classes'].get(), pretrained=False)
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    with torch.no_grad():
        predictions = model(image)

    boxes, labels, scores, masks = predictions[0]['boxes'].numpy(),\
                                   predictions[0]['labels'].numpy(), \
                                   predictions[0]['scores'].numpy(), \
                                   predictions[0]['masks'].numpy()

    return boxes, labels, scores, masks
