import asyncio
import logging
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
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
        self.current_loss: float = float("inf")
        self.best_loss: float = float("inf")
        self.status: str = "initializing"
        self.detail: str = ""
        self.output_logs: List[str] = []
        logger.debug("TrainingProgress initialized")

    def __str__(self):
        return (
            f"TrainingProgress(epoch={self.current_epoch + 1}/{self.total_epochs}, "
            f"loss={self.current_loss:.4f}, status={self.status})"
        )


class ProgressCallback(Callback):
    def __init__(self, progress: TrainingProgress, progress_callback=None, loop=None):
        super().__init__()
        self.progress = progress
        self.progress_callback = progress_callback
        self.loop = loop
        logger.info(f"ProgressCallback initialized with progress: {progress}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info(f"Epoch {trainer.current_epoch} completed")
        self.progress.current_epoch = trainer.current_epoch
        self.progress.current_loss = trainer.callback_metrics.get("train/loss_epoch", torch.tensor(float("inf"))).item()

        if self.progress.current_loss < self.progress.best_loss:
            self.progress.best_loss = self.progress.current_loss
            logger.debug(f"New best loss: {self.progress.best_loss:.4f}")

        self.progress.status = "training"
        self.progress.detail = f"Training epoch {self.progress.current_epoch + 1}/{self.progress.total_epochs}"
        logger.info(f"Progress updated in callback: {self.progress}")

        if self.progress_callback and self.loop:
            logger.info("Sending progress update")
            try:
                asyncio.run_coroutine_threadsafe(self.progress_callback(self.progress), self.loop)
                logger.info("Progress update task scheduled")
            except Exception as e:
                logger.error(f"Error scheduling progress update task: {str(e)}")


class TrainingService:
    def __init__(self):
        self.active_trainings: Dict[str, TrainingProgress] = {}
        logger.info("TrainingService initialized")

    def get_model_trainer(self, model_name: str):
        """Get the appropriate model trainer based on model name."""
        logger.info(f"Getting model trainer for: {model_name}")
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
        progress = TrainingProgress()
        try:
            logger.info(f"Starting training for experiment: {request.experiment_name}")
            # Initialize progress tracking
            self.active_trainings[request.experiment_name] = progress
            logger.info(f"Progress tracking initialized: {progress}")

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
            progress.status = "intializing"
            progress.current_epoch = -1
            progress.detail = "Training started"
            logger.info(f"Progress tracking set up: {progress}")

            event_loop = None
            if progress_callback:
                event_loop = asyncio.get_running_loop()
                logger.info("Calling initial progress callback")
                await progress_callback(progress)

                trainer.callbacks.append(ProgressCallback(progress, progress_callback, event_loop))
                logger.info("Progress callback added to trainer")

            # Start training
            logger.info("Starting training process")
            await asyncio.to_thread(trainer.train)
            logger.info("Training completed")

            # Training completed
            progress.status = "completed"
            progress.detail = "Training completed successfully"
            logger.info(f"Final progress: {progress}")
            if progress_callback:
                logger.info("Calling final progress callback")
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
            logger.info(f"Retrieved progress for {experiment_name}: {progress}")
        else:
            logger.debug(f"No progress found for experiment: {experiment_name}")
        return progress


training_service = TrainingService()
