import random
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms as T

from autovisionai.configs.config import CONFIG


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


def get_transform(resize: bool = False, random_crop: bool = False, hflip: bool = False) -> Compose:
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
                )
            )
        )
    elif random_crop:
        transforms.append(
            RandomCrop(
                prob=CONFIG["data_augmentation"]["random_crop_prob"].get(),
                crop_to=(
                    CONFIG["data_augmentation"]["random_crop_crop_to"].get(),
                    CONFIG["data_augmentation"]["random_crop_crop_to"].get(),
                ),
            )
        )
    if hflip:
        transforms.append(HorizontalFlip(prob=CONFIG["data_augmentation"]["h_flip_prob"].get()))

    return Compose(transforms)
