from autovisionai.configs.config import CONFIG
from autovisionai.processing.datamodule import CarsDataModule


datamodule = CarsDataModule(data_root=CONFIG['dataset']['data_root'].get(),
                            batch_size=2,
                            num_workers=0)
datamodule.setup()


def test_train_val_split_size():
    assert len(datamodule.data_train) == 1280
    assert len(datamodule.data_val) == 319


def test_train_dataloader_size():
    assert len(datamodule.train_dataloader()) == 640


def test_val_dataloader_size():
    assert len(datamodule.val_dataloader()) == 160
