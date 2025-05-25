import asyncio
import logging
from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from autovisionai.api.schemas.train import TrainingRequest
from autovisionai.core.configs import CONFIG
from autovisionai.core.models.fast_scnn.fast_scnn_trainer import FastSCNNTrainer
from autovisionai.core.models.mask_rcnn.mask_rcnn_trainer import MaskRCNNTrainer
from autovisionai.core.models.unet.unet_trainer import UnetTrainer
from autovisionai.core.train import ModelTrainer

logger = logging.getLogger(__name__)


class TrainingProgress:
    def __init__(self):
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.current_loss: float = 0.0
        self.best_loss: float = float("inf")
        self.status: str = "initializing"
        self.detail: str = ""


class ProgressCallback(Callback):
    def __init__(self, progress: TrainingProgress, progress_callback=None):
        super().__init__()
        self.progress = progress
        self.progress_callback = progress_callback

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info(f"Epoch {trainer.current_epoch} completed")
        self.progress.current_epoch = trainer.current_epoch
        self.progress.current_loss = trainer.callback_metrics.get("train/loss_epoch", 0.0)
        if self.progress.current_loss < self.progress.best_loss:
            self.progress.best_loss = self.progress.current_loss
        self.progress.status = "training"
        self.progress.detail = f"Training epoch {self.progress.current_epoch}/{self.progress.total_epochs}"
        logger.info(f"Progress updated: {self.progress.__dict__}")
        if self.progress_callback:
            asyncio.create_task(self.progress_callback(self.progress))


class TrainingService:
    def __init__(self):
        self.active_trainings: Dict[str, TrainingProgress] = {}

    def get_model_trainer(self, model_name: str):
        """Get the appropriate model trainer based on model name."""
        match model_name.lower():
            case "unet":
                return UnetTrainer()
            case "fast_scnn":
                return FastSCNNTrainer()
            case "mask_rcnn":
                return MaskRCNNTrainer()
            case _:
                raise ValueError(f"Unsupported model '{model_name}'.\nAvailable models: {CONFIG.models.available}.")

    async def train_model(self, request: TrainingRequest, progress_callback=None) -> Dict:
        """Train a model with progress updates."""
        try:
            logger.info(f"Starting training for experiment: {request.experiment_name}")
            # Initialize progress tracking
            progress = TrainingProgress()
            self.active_trainings[request.experiment_name] = progress
            logger.info(f"Progress tracking initialized: {progress.__dict__}")

            # Get model trainer
            model = self.get_model_trainer(request.model_name)
            logger.info(f"Model trainer created: {request.model_name}")

            # Initialize ModelTrainer
            trainer = ModelTrainer(
                experiment_name=request.experiment_name,
                model=model,
                batch_size=request.batch_size,
                epoch_patience=request.epoch_patience,
                use_resize=request.use_resize,
                use_random_crop=request.use_random_crop,
                use_hflip=request.use_hflip,
                max_epochs=request.max_epochs or CONFIG.trainer.max_epoch,
            )
            logger.info("ModelTrainer initialized")

            # Set up progress tracking
            progress.total_epochs = trainer.config.max_epochs
            progress.status = "training"
            progress.detail = "Training started"
            logger.info(f"Progress tracking set up: {progress.__dict__}")

            if progress_callback:
                await progress_callback(progress)

            # Add progress callback to trainer
            trainer.callbacks.append(ProgressCallback(progress, progress_callback))
            logger.info("Progress callback added to trainer")

            # Start training
            logger.info("Starting training process")
            trainer.train()
            logger.info("Training completed")

            # Training completed
            progress.status = "completed"
            progress.detail = "Training completed successfully"
            logger.info(f"Final progress: {progress.__dict__}")
            if progress_callback:
                await progress_callback(progress)

            return {
                "status": "success",
                "detail": "Training completed successfully",
                "experiment_path": str(trainer.experiment_path),
                "model_weights_path": str(trainer._save_model_weights()),
            }

        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
            progress.status = "error"
            progress.detail = str(e)
            if progress_callback:
                await progress_callback(progress)
            return {
                "status": "error",
                "detail": str(e),
                "experiment_path": None,
                "model_weights_path": None,
            }
        finally:
            if request.experiment_name in self.active_trainings:
                del self.active_trainings[request.experiment_name]
                logger.info(f"Progress tracking removed for experiment: {request.experiment_name}")

    def get_training_progress(self, experiment_name: str) -> Optional[TrainingProgress]:
        """Get the current progress of a training job."""
        progress = self.active_trainings.get(experiment_name)
        if progress:
            logger.debug(f"Retrieved progress for {experiment_name}: {progress.__dict__}")
        return progress


# Global training service instance
training_service = TrainingService()
