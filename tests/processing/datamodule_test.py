from autovisionai.configs.config import CONFIG
from autovisionai.processing.datamodule import CarsDataModule

datamodule = CarsDataModule(data_root=CONFIG['dataset']['test_data_root'].get(),
                            batch_size=2,
                            num_workers=2)
datamodule.setup()


def test_train_val_split_size():
    assert len(datamodule.data_train) == 13
    assert len(datamodule.data_val) == 3


def test_train_dataloader_size():
    assert len(datamodule.train_dataloader()) == 7


def test_val_dataloader_size():
    assert len(datamodule.val_dataloader()) == 2
