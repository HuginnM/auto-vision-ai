import torch
from numpy.testing import assert_almost_equal

from autovisionai.core.models.fast_scnn.fast_scnn_trainer import FastSCNNTrainer


def set_seed(seed):
    torch.manual_seed(seed)


def generate_test_batch():
    test_annotations = tuple([{"mask": torch.randint(0, 1, (1, 512, 512), dtype=torch.uint8)} for _ in range(4)])
    test_images = tuple([torch.randn((3, 512, 512)) for _ in range(4)])
    test_batch = (test_images, test_annotations)
    return test_batch


def test_training_step():
    set_seed(42)

    test_batch = generate_test_batch()

    model = FastSCNNTrainer(1)
    outputs = model.training_step(test_batch, 0)

    assert_almost_equal(outputs.item(), 0.6832375526428223, decimal=5)


def test_validation_step():
    set_seed(42)

    test_batch = generate_test_batch()

    model = FastSCNNTrainer(1)
    outputs = model.validation_step(test_batch, 0)

    assert_almost_equal(outputs["val_loss"].item(), 0.6832375526428223, decimal=5)
    assert_almost_equal(outputs["val_iou"].item(), 0.27531957626342773, decimal=5)
