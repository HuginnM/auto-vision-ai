import os
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from autovisionai.configs.config import CONFIG
from autovisionai.processing.datamodule import CarsDataModule

accelerator = "gpu" if torch.cuda.is_available() else "cpu"


def train_model(
    exp_number: int,
    model: Any,
    batch_size: int = 4,
    max_epochs: int = 1,
    epoch_patience: int = 2,
    use_resize: bool = False,
    use_random_crop: bool = False,
    use_hflip: bool = False,
) -> None:
    """
    Trains a Segmentation model based on a given config.

    :param exp_number: number of training experiment.
    :param model: an instance of the model we are going to train
    :param batch_size: a number of samples per batch.
    :param max_epochs: a number of epochs to train model.
    :param epoch_patience: a number of epochs without improvement for early stopping.
    :param use_resize: specify, whether to apply image resize or not.
    :param use_random_crop: specify, whether to apply image resize and random crop or not.
    :param use_hflip: specify, whether to apply image horizontal flip or not.
    """
    datamodule = CarsDataModule(
        data_root=CONFIG["dataset"]["data_root"].get(),
        batch_size=batch_size,
        num_workers=CONFIG["dataloader"]["num_workers"].get(),
        resize=use_resize,
        random_crop=use_random_crop,
        hflip=use_hflip,
    )

    experiment_folder = "exp_" + str(exp_number)

    # Creates experiment folders to save there logs and weights
    # Weights folder:
    weights_folder_path = os.path.join(
        os.path.join(CONFIG["trainer"]["logs_and_weights_root"].get(), experiment_folder),
        CONFIG["trainer"]["weights_folder"].get(),
    )
    os.makedirs(weights_folder_path, exist_ok=True)
    # Logs folder:
    logs_folder_path = os.path.join(
        os.path.join(CONFIG["trainer"]["logs_and_weights_root"].get(), experiment_folder),
        CONFIG["trainer"]["logger_folder"].get(),
    )
    os.makedirs(logs_folder_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_folder_path,
        every_n_epochs=1,
        monitor="val/loss_epoch",
        auto_insert_metric_name=False,
        filename="sample-cars-segm-model-epoch{epoch:02d}-val_loss_epoch{val/loss_epoch:.3f}",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/loss_epoch",
        patience=epoch_patience,  # stop after N epochs without improvement
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=TensorBoardLogger(
            save_dir=os.path.join(CONFIG["trainer"]["logs_and_weights_root"].get(), experiment_folder),
            name=CONFIG["trainer"]["logger_folder"].get(),
        ),
        log_every_n_steps=CONFIG["trainer"]["log_every_n_steps"].get(),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, datamodule)

    torch.save(model.model.state_dict(), os.path.join(weights_folder_path, "model.pt"))


if __name__ == "__main__":
    from autovisionai.models.unet.unet_trainer import UnetTrainer

    model = UnetTrainer()
    train_model(
        exp_number=1, model=model, batch_size=4, max_epochs=5, use_resize=False, use_random_crop=True, use_hflip=True
    )
