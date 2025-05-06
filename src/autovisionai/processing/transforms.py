import random
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms as T

from autovisionai.configs.config import CONFIG
from autovisionai.loggers.app_logger import logger
from autovisionai.utils.utils import find_bounding_box


class ToTensor:
    """
    Transforms the PIL.Image into torch.Tensor.
    """

    def __call__(
        self, image: Image.Image, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param image: a PIL.Image object.
        :param target: annotation with target image_id and mask.
        :return: a converted torch.Tensor image and its target annotation.
        """
        image = T.ToTensor()(image)
        return image, target


class Resize:
    """
    Resizes image and ground truth mask.

    :param resize_to: a size of image and mask to be resized to.
    """

    def __init__(self, resize_to: Tuple[int, int]):
        self.resize_to = resize_to

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]):
        """
        :param image: a torch.Tensor image.
        :param target: a dict of target annotations.
        :return: a resized image and target annotations with resized mask.
        """
        image = F.interpolate(image.unsqueeze(0), size=self.resize_to, mode="bilinear", align_corners=False)
        mask = F.interpolate(target["mask"].float().unsqueeze(0), size=self.resize_to, mode="nearest-exact")

        image = image.squeeze(0)
        mask = mask.squeeze(0).to(torch.uint8)
        target["mask"] = mask

        return image, target


class RandomCrop:
    """
    Randomly crop the image and mask with a given probability.
    :param prob: a probability of image and mask being cropped.
    :param crop_to: a size of image and mask to be cropped to.
    """

    def __init__(self, prob: float, crop_to: Tuple[int, int]) -> None:
        self.prob = prob
        self.crop_to = crop_to

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param image: a torch.Tensor image.
        :param target: a dict of target annotations.
        :return: cropped image and target annotation with cropped mask.
        """
        if random.random() < self.prob:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop_to)

            image = TF.crop(image, i, j, h, w)  # image[:, i:i+h, j:j+w]
            target["mask"] = TF.crop(target["mask"], i, j, h, w)  # target['mask'][:, i:i+h, j:j+w]
        else:
            image, target = Resize(self.crop_to)(image, target)

        return image, target


class RandomCropWithObject:
    """
    Randomly crops the image and mask, ensuring the object is not completely lost (mask not empty).
    Falls back to Resize if all attempts fail.

    :param prob: Probability to apply random crop.
    :param crop_to: Output size (H, W).
    :param max_tries: How many times to try finding a crop with some object inside.
    """

    def __init__(self, prob: float, crop_to: Tuple[int, int], add_bbox: bool = False, max_tries: int = 10) -> None:
        self.prob = prob
        self.crop_to = crop_to
        self.max_tries = max_tries
        self.add_bbox = add_bbox
        self.resizer = Resize(crop_to)  # fallback resizer

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if random.random() >= self.prob:
            return self.resizer(image, target)

        for _ in range(self.max_tries):
            i, j, h_crop, w_crop = T.RandomCrop.get_params(image, output_size=self.crop_to)
            cropped_mask = TF.crop(target["mask"], i, j, h_crop, w_crop)

            if torch.any(cropped_mask):
                bbox_of_mask = find_bounding_box(cropped_mask)

                # If bbox is valid, then find_bounding_box will return
                # valid bbox tensor and the model can use it.
                if isinstance(bbox_of_mask, torch.Tensor):
                    image = TF.crop(image, i, j, h_crop, w_crop)
                    target["mask"] = cropped_mask

                    if self.add_bbox:
                        target["box"] = bbox_of_mask

                    # logger.debug(f"Applied crop (attempt {attempt + 1}) at (i={i}, j={j})")
                    return image, target

        logger.warning("Fallback: all crops were empty — applied resize")
        return self.resizer(image, target)


class AddBoundingBox:
    """
    Adds box field to the target. If the target already has 'box'
    (e.g. after RandomCropWithObject) -- do nothing and returns input.

    """

    def __call__(self, image, target):
        if "box" not in target:
            target["box"] = find_bounding_box(target["mask"])

        return image, target


class HorizontalFlip:
    """
    Horizontally flips the image and mask with a given probability.

    :param prob: a probability of image and mask being flipped.
    """

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param image: a torch.Tensor image.
        :param target: a dict of target annotation.
        :return: a horizontally flipped image and target annotation with flipped mask.
        """
        if random.random() < self.prob:
            image = TF.hflip(image)
            target["mask"] = TF.hflip(target["mask"])

        return image, target


class Compose:
    """
    Composes several transforms together.
    :param transforms: a list of transforms to compose.
    """

    def __init__(self, transforms: Optional[list] = None) -> None:
        self.transforms = transforms

    def __call__(
        self, image: Union[Image.Image, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param image: an image to be transformed.
        :param target: a dict of target annotation.
        :return: a transformed image and target annotation with transformed mask.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transform(resize: bool = False, random_crop: bool = False, hflip: bool = False, bbox: bool = False) -> Compose:
    """
    Compose transforms for image amd mask augmentations.
    :param resize: specify, whether to apply image resize or not.
    :param random_crop: specify, whether to apply image random crop or not.
    :param hflip: specify, whether to apply image horizontal flipped or not.
    :return: composed transforms.
    """
    transforms = [ToTensor()]

    if resize:
        transforms.append(
            Resize(
                resize_to=(
                    CONFIG["data_augmentation"]["resize_to"].get(),
                    CONFIG["data_augmentation"]["resize_to"].get(),
                ),
            )
        )
        logger.info("Added Resize transformation.")
    elif random_crop:
        transforms.append(
            RandomCropWithObject(
                prob=CONFIG["data_augmentation"]["random_crop_prob"].get(),
                crop_to=(
                    CONFIG["data_augmentation"]["random_crop_crop_to"].get(),
                    CONFIG["data_augmentation"]["random_crop_crop_to"].get(),
                ),
                add_bbox=bbox,
            )
        )
        logger.info("Added RandomCropWithObject transformation.")
    if hflip:
        transforms.append(HorizontalFlip(prob=CONFIG["data_augmentation"]["h_flip_prob"].get()))
        logger.info("Added HorizontalFlip transformation.")
    if bbox:
        transforms.append(AddBoundingBox())
        logger.info("Added AddBoundingBox transformation.")

    return Compose(transforms)
