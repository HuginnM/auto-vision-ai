import asyncio
from typing import Dict, Optional

import pytorch_lightning as pl

from autovisionai.api.schemas.train import TrainingRequest
from autovisionai.core.configs import CONFIG
from autovisionai.core.models.fast_scnn.fast_scnn_trainer import FastSCNNTrainer
from autovisionai.core.models.mask_rcnn.mask_rcnn_trainer import MaskRCNNTrainer
from autovisionai.core.models.unet.unet_trainer import UnetTrainer
from autovisionai.core.train import ModelTrainer


class TrainingProgress:
    def __init__(self):
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.current_loss: float = 0.0
        self.best_loss: float = float("inf")
        self.status: str = "initializing"
        self.detail: str = ""


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
            # Initialize progress tracking
            progress = TrainingProgress()
            self.active_trainings[request.experiment_name] = progress

            # Get model trainer
            model = self.get_model_trainer(request.model_name)

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

            # Set up progress tracking
            progress.total_epochs = trainer.config.max_epochs
            progress.status = "training"
            progress.detail = "Training started"

            if progress_callback:
                await progress_callback(progress)

            # Add progress callback to trainer
            def on_epoch_end(trainer, *args, **kwargs):
                progress.current_epoch = trainer.current_epoch
                progress.current_loss = trainer.callback_metrics.get("train/loss_epoch", 0.0)
                if progress.current_loss < progress.best_loss:
                    progress.best_loss = progress.current_loss
                if progress_callback:
                    asyncio.create_task(progress_callback(progress))

            trainer.callbacks.append(pl.callbacks.Callback(on_epoch_end=on_epoch_end))

            # Start training
            trainer.train()

            # Training completed
            progress.status = "completed"
            progress.detail = "Training completed successfully"
            if progress_callback:
                await progress_callback(progress)

            return {
                "status": "success",
                "detail": "Training completed successfully",
                "experiment_path": str(trainer.experiment_path),
                "model_weights_path": str(trainer._save_model_weights()),
            }

        except Exception as e:
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

    def get_training_progress(self, experiment_name: str) -> Optional[TrainingProgress]:
        """Get the current progress of a training job."""
        return self.active_trainings.get(experiment_name)


# Global training service instance
training_service = TrainingService()
