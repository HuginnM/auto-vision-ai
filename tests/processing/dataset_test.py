import torch
from PIL import Image

from autovisionai.configs.config import CONFIG
from autovisionai.processing.dataset import CarsDataset

dataset = CarsDataset(data_root=CONFIG['dataset']['data_root'].get())
image, annotation = dataset[10]


def test_image_type():
    assert isinstance(image, Image.Image)


def test_image_id_dtype():
    assert annotation['image_id'].dtype == torch.int64


def test_mask_dtype():
    assert annotation['mask'].dtype == torch.uint8
    assert (annotation['mask'].unique() == torch.tensor([0, 1], dtype=torch.uint8)).all()


def test_mask_shape():
    assert annotation['mask'].shape == torch.Size([1, 1280, 1918])


def test_mask_values():
    assert (annotation['mask'][:, 1000, 925:935] == torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                                                 dtype=torch.uint8)).all()


def test_len_dataset():
    assert dataset.__len__() == 1600