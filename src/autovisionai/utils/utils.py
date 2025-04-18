import os
from io import BytesIO
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

from autovisionai.configs.config import CONFIG


def show_pic_and_original_mask(image_id: int) -> None:
    """
    Helper function to plot original image, original mask and bitwise picture.

    :param image_id: an image id.
    """
    imgs_list = sorted(
        os.listdir(os.path.join(CONFIG['dataset']['data_root'].get(), CONFIG['dataset']['images_folder'].get())))
    masks_list = sorted(os.listdir(
        os.path.join(CONFIG['dataset']['data_root'].get(), CONFIG['dataset']['masks_folder'].get())))

    img_path = os.path.join(CONFIG['dataset']['data_root'].get(), CONFIG['dataset']['images_folder'].get(),
                            imgs_list[image_id])
    mask_path = os.path.join(CONFIG['dataset']['data_root'].get(), CONFIG['dataset']['masks_folder'].get(),
                             masks_list[image_id])

    img = Image.open(img_path).convert('RGB')
    img = np.asarray(img)

    mask = Image.open(mask_path)
    mask = np.array(mask)

    # get masked value (foreground)
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    # add the 3d dim to mask and convert mask values to white color
    mask3 = np.stack([mask, mask, mask]).transpose((1, 2, 0))
    np.putmask(mask3, mask3 > 0, 255)

    # concatenate images Horizontally
    horizontal_imgs = np.concatenate((img, img_masked, mask3), axis=1)

    try:
        cv2.imshow('Original Image | Original Image + Mask | Mask', horizontal_imgs)
        cv2.waitKey(0)
    except Exception as e:
        print('Error:', e)
        plt.figure(figsize=(40, 10))
        plt.imshow(horizontal_imgs)
        plt.title('Original Image | Original Image + Mask | Mask', fontsize=50)


def show_pic_and_pred_semantic_mask(img: torch.Tensor, pred_mask: np.ndarray,
                                    threshold: float = 0.5, use_plt: bool = False) -> None:
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
            cv2.imshow('Original Image with Predicted Semantic Mask', img)
            cv2.waitKey(0)
        except Exception as e:
            print('Error:', e)
            plt.figure(figsize=(20, 10))
            plt.imshow(img)
            plt.title('Original Image with Predicted Semantic Mask', fontsize=40)


def show_pic_and_pred_instance_masks(img: torch.Tensor, pred_masks: np.ndarray,
                                     scores: np.ndarray, min_score: float = 0.8,
                                     threshold: float = 0.5, use_plt: bool = True) -> None:
    """
    Helper function to plot original image and predicted instance masks.

    :param img: an image for which the trained model predicts segmentation masks.
    :param pred_masks: predicted instance segmentation masks.
    :param scores: predicted scores.
    :param min_score: a min score to sort segmentation masks.
    :param threshold: a min score for predicted mask pixel after using sigmoid func. Values from 0 to 1.
    :param use_plt: show image with plt for better experience with JN when True. Instead shows it via GUI with cv2.
    """
    img = np.asarray(img * 255, dtype='uint8').squeeze()
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
            cv2.imshow('Original Image with Predicted Instances', img)
            cv2.waitKey(0)
        except Exception as e:
            print('Error:', e)
            plt.figure(figsize=(20, 10))
            plt.imshow(img)
            plt.title('Original Image with Predicted Instances', fontsize=40)


def get_input_image_for_inference(local_path: str = None, url: str = None) -> torch.Tensor:
    """
    Converts image into tensor [1, 3, H, W] for model inference.

    :param local_path: a local path to the image.
    :param url: an image url.
    :return: a torch tensor with shape [1, 3, H, W].
    """
    if local_path is not None:
        img = Image.open(local_path).convert('RGB')
    elif url is not None:
        # adding this header to avoid some sites blocking. Imitating user behaviour.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # add more protections
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        raise ValueError("Provide either `local_path` or `url`.")
        raise
    img = F.to_tensor(img)
    img = img.unsqueeze(0)
    return img


def get_batch_images_and_pred_masks_in_a_grid(eval_step_output: Union[List[Dict[str, torch.Tensor]], torch.Tensor],
                                              images: Tuple[torch.Tensor, ...],
                                              mask_rcnn: bool = False) -> torch.Tensor:
    """
    Makes a grid of images and their predicted masks on validation_step.

    :param mask_rcnn: specifies whether eval_step_output is a Mask RCNN output or not.
    :param eval_step_output: an validation step output.
    :param images: a batch of images.
    :return: a tensor containing grid of images.
    """
    if mask_rcnn:
        # get top scored mask for each image in a batch
        # use sigmoid func on predicted masks and use threshold = 0.65
        pred_masks = torch.stack([dict_i['masks'][0].sigmoid().detach().cpu() > 0.65 for dict_i in eval_step_output])
    else:
        # use sigmoid func on predicted masks and use threshold = 0.5
        pred_masks = torch.stack([mask.sigmoid().detach().cpu() > 0.5 for mask in eval_step_output])

    # draw masks on images
    masks_on_images = torch.stack(
        [torchvision.utils.draw_segmentation_masks(image=images[i].detach().cpu().mul(255).type(torch.uint8),
                                                   masks=pred_masks[i].type(torch.bool),
                                                   alpha=0.8,
                                                   colors='blue') for i in range(len(images))])
    # make grid with predicted masks on batch images
    grid = torchvision.utils.make_grid(masks_on_images)

    return grid


def bboxes_iou(target: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Calculates an Intersection Over Union metric over bboxes.

    :param target: a dict with target annotations.
    :param pred: a dict with predicted annotations.
    :return: an IoU over bboxes score.
    """
    if pred['boxes'].shape[0] == 0:
        iou_score = torch.tensor(0.0, device=pred['boxes'].device)
        return iou_score
    iou_score = torchvision.ops.box_iou(target['boxes'], pred['boxes']).diag().mean()
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
    pred_masks = torch.where(probs > 0.5, 1., 0.)
    pred_masks = pred_masks.cpu()

    # def define_task(num_classes):
    #     if num_classes < 3:
    #         print('IoT: Binary task.')
    #         return 'binary'
    #     else:
    #         print('IoT: Multiclass task.')
    #         return 'multiclass'

    iou_score = jaccard_index(preds=pred_masks, target=target.squeeze(1).cpu(),
                              num_classes=num_classes, task='multiclass')  #task=define_task(num_classes))

    return iou_score
