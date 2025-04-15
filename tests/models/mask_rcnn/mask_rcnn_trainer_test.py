import torch
from numpy.testing import assert_almost_equal

from autovisionai.configs.config import CONFIG
from autovisionai.processing.datamodule import CarsDataModule
from autovisionai.models.mask_rcnn.mask_rcnn_trainer import MaskRCNNTrainer


def set_seed(seed):
    torch.manual_seed(seed)


def generate_test_batch():
    dm = CarsDataModule(data_root=CONFIG['dataset']['data_root'].get(), batch_size=2, num_workers=2)
    dm.setup()
    test_batch = next(iter(dm.train_dataloader()))
    return test_batch


def test_convert_targets_to_mask_rcnn_format():
    set_seed(42)
    test_batch = generate_test_batch()

    model = MaskRCNNTrainer()
    mask_rcnn_target = model._convert_targets_to_mask_rcnn_format(test_batch[1])

    assert list(mask_rcnn_target[0].keys()) == ['image_id', 'boxes', 'masks', 'labels']
    assert (mask_rcnn_target[0]['boxes'] == torch.tensor([[ 588.,  440., 1251., 1050.]], 
                                                         dtype=torch.float32)).all()


def test_step():
    set_seed(42)
    test_batch = generate_test_batch()

    model = MaskRCNNTrainer()
    outputs = model.step(test_batch, is_training=True)
    assert_almost_equal(outputs['loss'].item(), 3.9373013973236084, decimal=2)
    assert_almost_equal(outputs['loss_step'].item(), 3.9373013973236084, decimal=2)
    assert_almost_equal(outputs['loss_mask'].item(), 3.074519634246826, decimal=2)


def test_training_step():
    set_seed(42)
    test_batch = generate_test_batch()

    model = MaskRCNNTrainer()
    outputs = model.training_step(test_batch, 0)

    assert_almost_equal(outputs['loss'].item(), 3.9373013973236084, decimal=2)
    assert_almost_equal(outputs['loss_step'].item(), 3.9373013973236084, decimal=2)
    assert_almost_equal(outputs['loss_mask'].item(), 3.074519634246826, decimal=2)


def test_validation_step():
    set_seed(42)

    test_batch = generate_test_batch()

    model = MaskRCNNTrainer()
    outputs = model.validation_step(test_batch, 0)

    assert_almost_equal(outputs['val_outputs']['loss'].item(), 3.9373013973236084, decimal=2)
    assert_almost_equal(outputs['val_outputs']['loss_step'].item(), 3.9373013973236084, decimal=2)
    assert_almost_equal(outputs['val_outputs']['loss_mask'].item(), 3.074519634246826, decimal=2)
    assert_almost_equal(outputs['val_iou'].item(), 0.25738221406936646, decimal=2)
