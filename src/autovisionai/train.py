import logging
import traceback
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from autovisionai.configs import CONFIG
from autovisionai.loggers.ml_logging import (
    create_experiments_dirs,
    get_loggers,
    get_run_name,
    save_config_to_experiment,
)
from autovisionai.processing.datamodule import CarsDataModule

logger = logging.get_logger(__name__)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"


def train_model(
    experiment_name: str,
    model: Any,
    batch_size: int = 4,
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
    run_name = get_run_name()
    model_name = model._get_name()
    logger.info(
        f"Start training model ({model_name}) under experiment: '{experiment_name}' and run name: '{run_name}'."
    )

    datamodule = CarsDataModule(
        data_root=CONFIG.dataset.data_root,
        batch_size=batch_size,
        num_workers=CONFIG.dataloader.num_workers,
        resize=use_resize,
        random_crop=use_random_crop,
        hflip=use_hflip,
        bbox=True if isinstance(model, MaskRCNNTrainer) else False,
    )
    logger.info(f"Created datamodule with resize: {use_resize}, random_crop: {use_random_crop}, hflip: {use_hflip}.")

    experiment_folder = "exp_" + experiment_name
    experiment_path = CONFIG.logging.ml_loggers.root_dir / experiment_folder

    exp_paths = create_experiments_dirs(experiment_path, model_name, run_name)  # create logging folders and weights
    save_config_to_experiment(experiment_path)  # copy config to exp for reproducibility
    loggers = get_loggers(experiment_name, experiment_path, run_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_paths["weights_path"],
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
        max_epochs=CONFIG.trainer.max_epoch,
        accelerator=accelerator,
        devices=1,
        logger=loggers,
        log_every_n_steps=CONFIG.trainer.log_every_n_steps,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, datamodule)
    logger.info(f"The model {model_name} finished their training.")

    model_weights_path = exp_paths["weights_path"] / "model.pt"
    torch.save(model.model.state_dict(), model_weights_path)
    logger.info(f"The model {model_name} weights were saved to the folder: '{model_weights_path}'.")


if __name__ == "__main__":
    from autovisionai.models.fast_scnn.fast_scnn_trainer import FastSCNNTrainer
    from autovisionai.models.mask_rcnn.mask_rcnn_trainer import MaskRCNNTrainer
    from autovisionai.models.unet.unet_trainer import UnetTrainer

    models = [UnetTrainer, FastSCNNTrainer, MaskRCNNTrainer]

    for model in models:
        try:
            model = model()
            train_model(
                experiment_name="compare_all_models",
                model=model,
                batch_size=4,
                use_resize=False,
                use_random_crop=True,
                use_hflip=True,
            )
        except Exception:
            error_message = traceback.format_exc()
            logger.exception(f"For the model {model._get_name()} the training was unsuccessfull.\n", error_message)
