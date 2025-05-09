"""Training pipeline for AutoVisionAI models.

This module implements a professional training pipeline for various computer vision models
including UNet, FastSCNN, and Mask R-CNN. It provides a robust training infrastructure
with support for multiple logging backends, model checkpointing, and early stopping.

Typical usage:
    ```python
    trainer = ModelTrainer(
        experiment_name="car_segmentation",
        model=UnetTrainer(),
        batch_size=4,
        use_augmentation=True
    )
    trainer.train()
    ```
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from autovisionai.configs import CONFIG
from autovisionai.loggers.ml_logging import (
    create_experiments_dirs,
    get_loggers,
    get_run_name,
    log_model_artifacts,
    save_config_to_experiment,
)
from autovisionai.models.mask_rcnn.mask_rcnn_trainer import MaskRCNNTrainer
from autovisionai.processing.datamodule import CarsDataModule

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        experiment_name: Name of the training experiment
        batch_size: Number of samples per batch
        epoch_patience: Number of epochs without improvement for early stopping
        use_resize: Whether to apply image resizing
        use_random_crop: Whether to apply random cropping
        use_hflip: Whether to apply horizontal flipping
        max_epochs: Maximum number of training epochs
        log_every_n_steps: Frequency of logging steps
    """

    experiment_name: str
    batch_size: int = 4
    epoch_patience: int = 2
    use_resize: bool = False
    use_random_crop: bool = False
    use_hflip: bool = False
    max_epochs: int = CONFIG.trainer.max_epoch
    log_every_n_steps: int = CONFIG.trainer.log_every_n_steps


class ModelTrainer:
    """Professional training pipeline for AutoVisionAI models.

    This class implements a robust training pipeline with support for:
    - Multiple model architectures (UNet, FastSCNN, Mask R-CNN)
    - Comprehensive logging (TensorBoard, MLflow, Weights & Biases)
    - Model checkpointing and early stopping
    - Data augmentation options
    - GPU/CPU training

    Attributes:
        config: Training configuration
        model: PyTorch Lightning model to train
        device: Training device (GPU/CPU)
        experiment_path: Path to experiment directory
        loggers: List of active ML loggers
        callbacks: List of training callbacks
    """

    def __init__(
        self,
        experiment_name: str,
        model: pl.LightningModule,
        batch_size: int = 4,
        epoch_patience: int = 2,
        use_resize: bool = False,
        use_random_crop: bool = False,
        use_hflip: bool = False,
        max_epochs: int = CONFIG.trainer.max_epoch,
        datamodule: Optional[CarsDataModule] = None,
    ) -> None:
        """Initialize the model trainer.

        Args:
            experiment_name: Name of the training experiment
            model: PyTorch Lightning model to train
            batch_size: Number of samples per batch
            epoch_patience: Number of epochs without improvement for early stopping
            use_resize: Whether to apply image resizing
            use_random_crop: Whether to apply random cropping
            use_hflip: Whether to apply horizontal flipping
            max_epochs: Maximum number of training epochs (defaults to CONFIG.trainer.max_epoch)
            datamodule: Optional data module for testing purposes
        """
        self.config = TrainingConfig(
            experiment_name=experiment_name,
            batch_size=batch_size,
            epoch_patience=epoch_patience,
            use_resize=use_resize,
            use_random_crop=use_random_crop,
            use_hflip=use_hflip,
            max_epochs=max_epochs,
        )
        self.model = model
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self._datamodule = datamodule
        self._setup_training_environment()

    def _setup_training_environment(self) -> None:
        """Set up the training environment including directories and loggers."""
        self.run_name = get_run_name()
        self.model_name = self.model.model_name
        self._setup_experiment_directories()
        self._setup_loggers()
        self._setup_callbacks()

    def _setup_experiment_directories(self) -> None:
        """Set up experiment directories and save configuration."""
        self.experiment_folder = f"exp_{self.config.experiment_name}"
        self.experiment_path = CONFIG.logging.ml_loggers.root_dir / self.experiment_folder
        self.exp_paths = create_experiments_dirs(self.experiment_path, self.model_name, self.run_name)
        save_config_to_experiment(self.experiment_path)

    def _setup_loggers(self) -> None:
        """Set up logging backends."""
        self.ml_loggers = get_loggers(self.config.experiment_name, self.experiment_path, self.run_name)

    def _setup_callbacks(self) -> None:
        """Set up training callbacks."""
        self.callbacks: List[Callback] = [
            ModelCheckpoint(
                dirpath=self.exp_paths["weights_path"],
                every_n_epochs=1,
                monitor="val/loss_epoch",
                auto_insert_metric_name=False,
                filename=f"{self.model_name}-epoch{{epoch:02d}}-val_loss_epoch{{val/loss_epoch:.3f}}",
            ),
            EarlyStopping(
                monitor="val/loss_epoch",
                patience=self.config.epoch_patience,
                mode="min",
                verbose=True,
            ),
        ]

    def _create_datamodule(self) -> CarsDataModule:
        """Create and configure the data module.

        Returns:
            Configured CarsDataModule instance
        """
        if self._datamodule is not None:
            return self._datamodule

        return CarsDataModule(
            data_root=CONFIG.dataset.data_root,
            batch_size=self.config.batch_size,
            num_workers=CONFIG.dataloader.num_workers,
            resize=self.config.use_resize,
            random_crop=self.config.use_random_crop,
            hflip=self.config.use_hflip,
            bbox=isinstance(self.model, MaskRCNNTrainer),
        )

    def _save_model_weights(self) -> Path:
        """Save model weights to disk.

        Returns:
            Path to saved model weights
        """
        model_weights_path = self.exp_paths["weights_path"] / "model.pt"
        torch.save(self.model.model.state_dict(), model_weights_path)
        return model_weights_path

    def train(self) -> None:
        """Execute the training pipeline.

        This method:
        1. Creates and configures the data module
        2. Initializes the PyTorch Lightning trainer
        3. Trains the model
        4. Saves model weights
        5. Logs model weights to configured backends
        """
        try:
            # Log training start if we have loggers
            if self.ml_loggers:
                for ml_logger in self.ml_loggers:
                    ml_logger.log_hyperparams(self.config.__dict__)

            # Create data module
            datamodule = self._create_datamodule()

            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                accelerator=self.device,
                devices=1,
                logger=self.ml_loggers,
                log_every_n_steps=self.config.log_every_n_steps,
                callbacks=self.callbacks,
            )

            # Train model
            trainer.fit(self.model, datamodule)

            # Save and log model weights
            model_weights_path = self._save_model_weights()
            if self.ml_loggers:
                log_model_artifacts(self.ml_loggers, self.model_name, str(model_weights_path))

        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}") from e


def main() -> None:
    """Main entry point for training multiple models."""
    from autovisionai.models.fast_scnn.fast_scnn_trainer import FastSCNNTrainer

    # models = [UnetTrainer, FastSCNNTrainer, MaskRCNNTrainer]
    models = [FastSCNNTrainer]

    # Test datamodule with only 16 images for fast test-training
    # test_datamodule = CarsDataModule(
    #     data_root=CONFIG.dataset.test_data_root,
    #     batch_size=1,
    #     num_workers=1,
    #     resize=False,
    #     random_crop=True,
    #     hflip=True,
    # )
    for model_class in models:
        try:
            model = model_class()
            trainer = ModelTrainer(
                experiment_name="final_test",
                model=model,
                batch_size=4,
                use_resize=False,
                use_random_crop=True,
                use_hflip=True,
                max_epochs=1,
                # datamodule=test_datamodule
            )
            trainer.train()
        except Exception as e:
            logger.exception(f"Training failed for {model_class.__name__}: {str(e)}")


if __name__ == "__main__":
    main()
