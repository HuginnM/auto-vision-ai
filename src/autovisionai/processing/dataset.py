import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from autovisionai.configs.config import CONFIG


class CarsDataset(Dataset):
    """
    Dataset for training and validation data.
    :param data_root: a path, where data is stored.
    :param transforms: torchvision transforms.
    """

    def __init__(self, data_root: str, transforms: Optional[Any] = None) -> None:
        self.data_root = data_root
        self.transforms = transforms

        self.allowed_extensions = tuple(CONFIG["dataset"]["allowed_extensions"].get())

        imgs_dir = os.path.join(data_root, CONFIG["dataset"]["images_folder"].get())
        masks_dir = os.path.join(data_root, CONFIG["dataset"]["masks_folder"].get())

        self.imgs_list = self._get_valid_files(imgs_dir)
        self.masks_list = self._get_valid_files(masks_dir)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """
        Retrieve a sample from the dataset.
        :param idx: an index of the sample to retrieve.
        :return: a tuple containing an PIL image and a dict with target annotations data.
        """
        img_path = os.path.join(self.data_root, CONFIG["dataset"]["images_folder"].get(), self.imgs_list[idx])
        mask_path = os.path.join(self.data_root, CONFIG["dataset"]["masks_folder"].get(), self.masks_list[idx])

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask = torch.tensor(np.array(mask), dtype=torch.uint8).unsqueeze(0)
        image_id = torch.tensor([idx])
        target = {"image_id": image_id, "mask": mask}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        """
        Retrieve the number of samples in the dataset.
        :return: a len of dataset.
        """
        samples_number = len(self.imgs_list)
        return samples_number

    def _get_valid_files(self, directory: str) -> List[str]:
        """Returns sorted list of valid image filenames in a directory."""
        return sorted([f for f in os.listdir(directory) if f.lower().endswith(self.allowed_extensions)])
