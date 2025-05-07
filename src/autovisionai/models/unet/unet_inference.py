import logging

import numpy as np
import torch
from torchvision import transforms as T

from autovisionai.configs import CONFIG
from autovisionai.models.unet.unet_model import Unet

logger = logging.get_logger(__name__)


def model_inference(trained_model_path: str, image: torch.Tensor) -> np.ndarray:
    """
    Loads the entire trained UNet model and predicts segmentation mask for an input image.

    :param trained_model_path: a path to saved model.
    :param image: a torch tensor with a shape [1, 3, H, W].
    :return: predicted mask for an input image.
    """

    model = Unet(CONFIG.models.unet.in_channels, CONFIG.models.unet.n_classes)
    model.load_state_dict(torch.load(trained_model_path))
    logger.info(f"The UNET model has been loaded with trained params from {trained_model_path}.")
    model.eval()

    resize_to = (CONFIG.data_augmentation.resize_to, CONFIG.data_augmentation.resize_to)

    resized_image = T.Resize(resize_to)(image)

    with torch.no_grad():
        prediction = model(resized_image)
        mask = torch.sigmoid(prediction)

    orig_size = (image.shape[2], image.shape[3])
    resized_mask_to_original_image_size = T.Resize(orig_size)(mask)

    return resized_mask_to_original_image_size.numpy()
