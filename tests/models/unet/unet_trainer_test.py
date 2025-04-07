import torch
from numpy.testing import assert_almost_equal

from autovisionai.models.unet.unet_trainer import UnetTrainer


def set_seed(seed):
    torch.manual_seed(seed)


def generate_test_batch():
    test_annotations = tuple([{'mask': torch.randint(0, 1, (1, 32, 32), dtype=torch.uint8)} for _ in range(4)])
    test_images = tuple([torch.randn((3, 32, 32)) for _ in range(4)])
    test_batch = (test_images, test_annotations)
    return test_batch


def test_training_step():
    set_seed(42)

    test_batch = generate_test_batch()

    model = UnetTrainer(3, 1)
    outputs = model.training_step(test_batch, 0)
    assert_almost_equal(outputs.item(), 0.6361280679702759, decimal=1)


def test_validation_step():
    set_seed(42)

    test_batch = generate_test_batch()

    model = UnetTrainer(3, 1)
    outputs = model.validation_step(test_batch, 0)
    assert_almost_equal(outputs['val_loss'].item(), 0.6361280679702759, decimal=1)
    assert_almost_equal(outputs['val_iou'].item(), 0.3317837715148926, decimal=1)
