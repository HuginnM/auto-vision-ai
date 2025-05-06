from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from autovisionai.configs import CONFIG
from autovisionai.processing.dataset import CarsDataset
from autovisionai.processing.transforms import get_transform


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]]:
    """
    Defines how to collate batches.

    :param batch: a list containing tuples of images and annotations.
    :return: a collated batch. Tuple containing a tuple of images and a tuple of annotations.
    """
    return tuple(zip(*batch, strict=False))


class CarsDataModule(pl.LightningDataModule):
    """
    LightningDataModule to supply training and validation data.

    :param data_root: a path, where data is stored.
    :param batch_size: a number of samples per batch.
    :param num_workers: a number of subprocesses to use for data loading.
    :param resize: specify, whether to apply image and mask resize or not.
    :param random_crop: specify, whether to randomly crop image and mask or not.
    :param hflip: specify, whether to apply horizontal flip to image and mask or not.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int,
        resize: bool = False,
        random_crop: bool = False,
        hflip: bool = False,
        bbox: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.random_crop = random_crop
        self.hflip = hflip
        self.bbox = bbox
        self.full_data = None
        self.data_train = None
        self.data_val = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads in data from file and prepares PyTorch tensor datasets for train and val split.

        :param stage: an argument to separate setup logic for trainer.
        """
        if stage == "fit" or stage is None:
            transforms = get_transform(self.resize, self.random_crop, self.hflip, self.bbox)
            self.full_data = CarsDataset(self.data_root, transforms)

            n_sample = len(self.full_data)
            split_idx = round(n_sample * CONFIG.datamodule.training_set_size)

            self.data_train = Subset(self.full_data, range(n_sample)[:split_idx])
            self.data_val = Subset(self.full_data, range(n_sample)[split_idx:])

    def train_dataloader(self) -> DataLoader:
        """
        Represents a Python iterable over a train dataset.

        :return: a dataloader for training.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Represents a Python iterable over a validation dataset.

        :return: a dataloader for validation.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
        )
