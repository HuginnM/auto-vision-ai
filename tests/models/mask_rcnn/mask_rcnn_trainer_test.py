import torch
from numpy.testing import assert_almost_equal

from autovisionai.configs.config import CONFIG
from autovisionai.processing.datamodule import CarsDataModule
from autovisionai.models.mask_rcnn.mask_rcnn_trainer import MaskRCNNTrainer

accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'


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
                                                         dtype=torch.float32).to(accelerator)).all()
