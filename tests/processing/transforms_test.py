import numpy as np
import torch
from PIL import Image

from autovisionai.processing.transforms import HorizontalFlip, RandomCrop, Resize, ToTensor

NUMPY_IMAGE = np.array(
    [
        [
            [0.9378, 0.9137, 0.9059, 0.9137, 0.9333],
            [0.9412, 0.9120, 0.8941, 0.4020, 0.4392],
            [0.5467, 0.5516, 0.8833, 0.3859, 0.5804],
            [0.0935, 0.3098, 0.1137, 0.4859, 0.6655],
            [0.5698, 0.4394, 0.6000, 0.6431, 0.6922],
        ],
        [
            [0.9378, 0.9137, 0.9059, 0.9137, 0.9333],
            [0.9412, 0.9120, 0.8961, 0.4020, 0.4431],
            [0.5453, 0.5516, 0.8833, 0.3859, 0.6000],
            [0.0975, 0.3059, 0.0971, 0.4806, 0.6576],
            [0.5698, 0.4375, 0.6078, 0.6510, 0.7000],
        ],
        [
            [0.9300, 0.9059, 0.8980, 0.9059, 0.9333],
            [0.9333, 0.9041, 0.8824, 0.4098, 0.4608],
            [0.5573, 0.5437, 0.8755, 0.3859, 0.6157],
            [0.1053, 0.3275, 0.0853, 0.4475, 0.6400],
            [0.5620, 0.4120, 0.6000, 0.6392, 0.6961],
        ],
    ]
)

ANNOTATION = {
    "mask": torch.tensor(
        [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]], dtype=torch.uint8
    )
}


def set_seed(seed):
    torch.manual_seed(seed)


def test_to_tensor():
    image = Image.new(mode="RGB", size=(10, 10))
    tensor_image, _ = ToTensor()(image, ANNOTATION.copy())
    assert isinstance(tensor_image, torch.Tensor)


def test_resize():
    resized_image, resized_target = Resize((3, 3))(torch.as_tensor(NUMPY_IMAGE.copy()), ANNOTATION.copy())

    true_resized_image = torch.tensor(
        [
            [[0.9303, 0.9020, 0.7601], [0.5483, 0.8833, 0.5156], [0.4061, 0.4379, 0.6524]],
            [[0.9303, 0.9026, 0.7610], [0.5474, 0.8833, 0.5286], [0.4061, 0.4376, 0.6553]],
            [[0.9225, 0.8928, 0.7640], [0.5528, 0.8755, 0.5391], [0.4011, 0.4284, 0.6434]],
        ],
        dtype=torch.float64,
    )

    assert resized_image.shape == torch.Size([3, 3, 3])
    assert resized_target["mask"].shape == torch.Size([1, 3, 3])

    assert torch.allclose(true_resized_image, resized_image, atol=1e-01)
    assert (resized_target["mask"] == torch.tensor([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.uint8)).all()


def test_random_crop():
    set_seed(42)
    cropped_image, cropped_target = RandomCrop(1, (3, 3))(torch.as_tensor(NUMPY_IMAGE.copy()), ANNOTATION.copy())
    true_cropped_image = torch.tensor(
        [
            [[0.9059, 0.9137, 0.9333], [0.8941, 0.4020, 0.4392], [0.8833, 0.3859, 0.5804]],
            [[0.9059, 0.9137, 0.9333], [0.8961, 0.4020, 0.4431], [0.8833, 0.3859, 0.6000]],
            [[0.8980, 0.9059, 0.9333], [0.8824, 0.4098, 0.4608], [0.8755, 0.3859, 0.6157]],
        ],
        dtype=torch.float64,
    )

    assert cropped_image.shape == torch.Size([3, 3, 3])
    assert cropped_target["mask"].shape == torch.Size([1, 3, 3])

    assert torch.allclose(true_cropped_image, cropped_image, atol=1e-01)
    assert (cropped_target["mask"] == torch.tensor([[[0, 0, 0], [0, 0, 0], [1, 1, 0]]], dtype=torch.uint8)).all()

    resized_image, resized_target = RandomCrop(0, (2, 2))(torch.as_tensor(NUMPY_IMAGE.copy()), ANNOTATION.copy())

    true_resized_image = torch.tensor(
        [
            [[0.9194, 0.5381], [0.3098, 0.5619]],
            [[0.9194, 0.5389], [0.3080, 0.5595]],
            [[0.9115, 0.5451], [0.3163, 0.5351]],
        ],
        dtype=torch.float64,
    )

    assert resized_image.shape == torch.Size([3, 2, 2])
    assert resized_target["mask"].shape == torch.Size([1, 2, 2])

    assert torch.allclose(true_resized_image, resized_image, atol=1e-01)
    assert (resized_target["mask"] == torch.tensor([[[0, 0], [1, 0]]], dtype=torch.uint8)).all()


def test_horizontal_flip():
    h_flipped_image, h_flipped_target = HorizontalFlip(1)(torch.as_tensor(NUMPY_IMAGE.copy()), ANNOTATION.copy())

    assert (
        h_flipped_image
        == torch.tensor(
            [
                [
                    [0.9333, 0.9137, 0.9059, 0.9137, 0.9378],
                    [0.4392, 0.4020, 0.8941, 0.9120, 0.9412],
                    [0.5804, 0.3859, 0.8833, 0.5516, 0.5467],
                    [0.6655, 0.4859, 0.1137, 0.3098, 0.0935],
                    [0.6922, 0.6431, 0.6000, 0.4394, 0.5698],
                ],
                [
                    [0.9333, 0.9137, 0.9059, 0.9137, 0.9378],
                    [0.4431, 0.4020, 0.8961, 0.9120, 0.9412],
                    [0.6000, 0.3859, 0.8833, 0.5516, 0.5453],
                    [0.6576, 0.4806, 0.0971, 0.3059, 0.0975],
                    [0.7000, 0.6510, 0.6078, 0.4375, 0.5698],
                ],
                [
                    [0.9333, 0.9059, 0.8980, 0.9059, 0.9300],
                    [0.4608, 0.4098, 0.8824, 0.9041, 0.9333],
                    [0.6157, 0.3859, 0.8755, 0.5437, 0.5573],
                    [0.6400, 0.4475, 0.0853, 0.3275, 0.1053],
                    [0.6961, 0.6392, 0.6000, 0.4120, 0.5620],
                ],
            ],
            dtype=torch.float64,
        )
    ).all()

    assert (
        h_flipped_target["mask"]
        == torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.uint8
        )
    ).all()
