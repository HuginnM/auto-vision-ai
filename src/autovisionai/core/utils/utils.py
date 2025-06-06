import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision
from PIL import Image
from torchmetrics.functional import jaccard_index
from torchvision.transforms import functional as F
from torchvision.utils import save_image

from autovisionai.core.configs import CONFIG

logger = logging.getLogger(__name__)


def get_valid_files(dir_path: Path, allowed_extenstions: Union[List, Tuple]) -> List[str]:
    """Returns sorted list of valid image filenames in a directory."""
    return sorted(f.name for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in allowed_extenstions)


def show_pic_and_original_mask(image_id: int) -> None:
    """
    Helper function to plot original image, original mask and bitwise picture.

    :param image_id: an image id.
    """
    images_folder = CONFIG.dataset.data_root / CONFIG.dataset.images_folder
    masks_folder = CONFIG.dataset.data_root / CONFIG.dataset.masks_folder

    images_list = get_valid_files(images_folder, CONFIG.dataset.allowed_extensions)
    masks_list = get_valid_files(masks_folder, CONFIG.dataset.allowed_extensions)

    image_path = images_folder / images_list[image_id]
    mask_path = masks_folder / masks_list[image_id]

    img = Image.open(image_path).convert("RGB")
    img = np.asarray(img)

    mask = Image.open(mask_path)
    mask = np.array(mask)

    # get masked value (foreground)
    image_masked = cv2.bitwise_and(img, img, mask=mask)

    # add the 3d dim to mask and convert mask values to white color
    mask3 = np.stack([mask, mask, mask]).transpose((1, 2, 0))
    np.putmask(mask3, mask3 > 0, 255)

    # concatenate images Horizontally
    horizontal_imgs = np.concatenate((img, image_masked, mask3), axis=1)

    try:
        cv2.imshow("Original Image | Original Image + Mask | Mask", horizontal_imgs)
        cv2.waitKey(0)
    except Exception as e:
        print("Error:", e)
        plt.figure(figsize=(40, 10))
        plt.imshow(horizontal_imgs)
        plt.title("Original Image | Original Image + Mask | Mask", fontsize=50)


def show_pic_and_pred_semantic_mask(
    img: torch.Tensor, pred_mask: np.ndarray, threshold: float = 0.5, use_plt: bool = False
) -> None:
    """
    Helper function to plot original image and predicted semantic mask.

    :param img: an image for which the trained model predicts segmentation masks.
    :param pred_mask: predicted semantic mask.
    :param threshold: a min score for predicted mask pixel after using sigmoid func. Values from 0 to 1.
    :param use_plt: show image with plt for better experience with JN when True. Instead shows it via GUI with cv2.
    """
    img = np.asarray(img * 255, dtype="uint8").squeeze()
    img = img.transpose(1, 2, 0)

    mask = (pred_mask > threshold).squeeze()
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    color = list(np.random.choice(range(256), size=3))
    r[mask == 1], g[mask == 1], b[mask == 1] = color
    rgb_mask = np.stack([r, g, b], axis=2)
    img = cv2.addWeighted(img, 0.7, rgb_mask, 1, 0)

    if use_plt:
        # Jupyter-friendly visualization
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Original Image with Predicted Semantic Mask")
        plt.show()
    else:
        try:
            cv2.imshow("Original Image with Predicted Semantic Mask", img)
            cv2.waitKey(0)
        except Exception as e:
            print("Error:", e)
            plt.figure(figsize=(20, 10))
            plt.imshow(img)
            plt.title("Original Image with Predicted Semantic Mask", fontsize=40)


def show_pic_and_pred_instance_masks(
    img: torch.Tensor,
    pred_masks: np.ndarray,
    scores: np.ndarray,
    min_score: float = 0.8,
    threshold: float = 0.5,
    use_plt: bool = True,
) -> None:
    """
    Helper function to plot original image and predicted instance masks.

    :param img: an image for which the trained model predicts segmentation masks.
    :param pred_masks: predicted instance segmentation masks.
    :param scores: predicted scores.
    :param min_score: a min score to sort segmentation masks.
    :param threshold: a min score for predicted mask pixel after using sigmoid func. Values from 0 to 1.
    :param use_plt: show image with plt for better experience with JN when True. Instead shows it via GUI with cv2.
    """
    img = np.asarray(img * 255, dtype="uint8").squeeze()
    img = img.transpose(1, 2, 0)

    for mask, score in zip(pred_masks, scores, strict=False):
        if score > min_score:
            mask = (mask > threshold).squeeze()
            r = np.zeros_like(mask).astype(np.uint8)
            g = np.zeros_like(mask).astype(np.uint8)
            b = np.zeros_like(mask).astype(np.uint8)
            color = list(np.random.choice(range(256), size=3))
            r[mask == 1], g[mask == 1], b[mask == 1] = color
            rgb_mask = np.stack([r, g, b], axis=2)
            img = cv2.addWeighted(img, 1, rgb_mask, 0.8, 0)

    if use_plt:
        # Jupyter-friendly visualization
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Original Image with Predicted Semantic Mask")
        plt.show()
    else:
        try:
            cv2.imshow("Original Image with Predicted Instances", img)
            cv2.waitKey(0)
        except Exception as e:
            print("Error:", e)
            plt.figure(figsize=(20, 10))
            plt.imshow(img)
            plt.title("Original Image with Predicted Instances", fontsize=40)


def get_input_image_for_inference(local_path: str = None, url: str = None) -> torch.Tensor:
    """
    Converts image into tensor [1, 3, H, W] for model inference.

    :param local_path: a local path to the image.
    :param url: an image url.
    :return: a torch tensor with shape [1, 3, H, W].
    """
    if local_path is not None:
        img = Image.open(local_path).convert("RGB")
    elif url is not None:
        # adding this header to avoid some sites blocking. Imitating user behaviour.
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # add more protections
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        raise ValueError("Provide either `local_path` or `url`.")
        raise
    img = F.to_tensor(img)
    img = img.unsqueeze(0)
    return img


def get_batch_images_and_pred_masks_in_a_grid(
    eval_step_output: Union[List[Dict[str, torch.Tensor]], torch.Tensor],
    images: Tuple[torch.Tensor, ...],
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Makes a grid of images and their predicted masks on validation_step.

    :param mask_rcnn: specifies whether eval_step_output is a Mask RCNN output or not.
    :param eval_step_output: an validation step output.
    :param images: a batch of images.
    :return: a tensor containing grid of images.
    """
    pred_masks = torch.stack([mask.sigmoid().detach().cpu() > threshold for mask in eval_step_output])

    # draw masks on images
    masks_on_images = torch.stack(
        [
            torchvision.utils.draw_segmentation_masks(
                image=images[i].detach().cpu().mul(255).type(torch.uint8),
                masks=pred_masks[i].type(torch.bool),
                alpha=0.7,
                colors="blue",
            )
            for i in range(len(images))
        ]
    )
    # make grid with predicted masks on batch images
    grid = torchvision.utils.make_grid(masks_on_images, nrow=4)

    return grid


def bboxes_iou(target: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Calculates an Intersection Over Union metric over bboxes.

    :param target: a dict with target annotations.
    :param pred: a dict with predicted annotations.
    :return: an IoU over bboxes score.
    """
    if pred["boxes"].shape[0] == 0:
        iou_score = torch.tensor(0.0, device=pred["boxes"].device)
        return iou_score
    iou_score = torchvision.ops.box_iou(target["boxes"], pred["boxes"]).diag().mean()
    return iou_score


def masks_iou(target, preds, num_classes):
    """
    Calculates an Intersection Over Union metric over masks.

    :param target: a torch tensor with stacked target masks.
    :param preds: a prediction of the model.
    :param num_classes: a number of classes.
    :return: an IoU score over masks.
    """
    probs = torch.sigmoid(preds.squeeze(1))
    pred_masks = torch.where(probs > 0.5, 1.0, 0.0)
    pred_masks = pred_masks.cpu()
    iou_score = jaccard_index(
        preds=pred_masks, target=target.squeeze(1).cpu(), num_classes=num_classes, task="multiclass"
    )

    return iou_score


def save_tensor_image(tensor: torch.Tensor, filename: str, folder: str = "debug_failed_samples") -> None:
    """
    Saves a tensor image [C, H, W] to a PNG file in the specified folder.

    :param tensor: Tensor of shape [C, H, W], values in [0, 1] or [0, 255]
    :param filename: File name (e.g., 'bad_sample_001.png')
    :param folder: Folder to save into (default = 'debug_failed_samples')
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    tensor = tensor.detach().cpu().float()
    if tensor.max() > 1:
        tensor = tensor / 255.0

    save_image(tensor, folder_path / filename)


def find_bounding_box(mask, min_size: int = CONFIG.data_augmentation.bbox_min_size):
    if not torch.any(mask):
        print("The mask is empty. Returning the empty bbox.")
        return None

    mask = mask.squeeze(0)
    rows = torch.any(mask != 0, dim=1)
    cols = torch.any(mask != 0, dim=0)

    nz_rows = torch.where(rows)[0]
    nz_cols = torch.where(cols)[0]

    xmin = nz_cols[0].item()
    ymin = nz_rows[0].item()
    xmax = nz_cols[-1].item()
    ymax = nz_rows[-1].item()

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    if width < min_size or height < min_size:
        logger.debug("Boundary box is too small, returning empty BBOX.")
        return None

    bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

    return bbox
