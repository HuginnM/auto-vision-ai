import logging

import numpy as np
import torch
from torchvision import transforms as T

from autovisionai.core.configs import CONFIG
from autovisionai.core.models.fast_scnn.fast_scnn_model import FastSCNN

logger = logging.getLogger(__name__)


def model_inference(trained_model_path: str, image: torch.Tensor) -> np.ndarray:
    """
    Loads the entire trained Fast SCNN model and predicts segmentation mask for an input image.

    :param trained_model_path: a path to saved model.
    :param image: a torch tensor with a shape [1, 3, H, W].
    :return: predicted mask for an input image.
    """
    model = FastSCNN(CONFIG.models.fast_scnn.n_classes)
    model.load_state_dict(torch.load(trained_model_path))
    logger.info(f"The Fast SCNN model has been loaded with trained params from {trained_model_path}.")
    model.eval()

    resize_to = (CONFIG.data_augmentation.resize_to, CONFIG.data_augmentation.resize_to)

    resized_image = T.Resize(resize_to)(image)

    with torch.no_grad():
        prediction = model(resized_image)
        mask = torch.sigmoid(prediction[0])

    orig_size = (image.shape[2], image.shape[3])
    resized_mask_to_original_image_size = T.Resize(orig_size)(mask)

    return resized_mask_to_original_image_size.numpy()
