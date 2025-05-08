import datetime as dt
import io
import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict

import torch
import torchvision.transforms.functional as F
import wandb
from PIL import Image
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger

from autovisionai.configs import CONFIG, CONFIG_DIR, MLLoggersConfig

logger = logging.getLogger(__name__)


def get_run_name():
    local_tz = dt.datetime.now().astimezone().tzinfo
    return f"run_{dt.datetime.now(tz=local_tz).strftime('%Y%m%dT%H%M%SUTC%z')}".replace("+", "")  # ISO8601 format


def get_loggers(experiment_name: str, experiment_path: Path, run_name: str = "run_default") -> list:
    """
    Creates a list of loggers based on the provided config.
    Supports TensorBoard, MLflow, and Weights & Biases (W&B).
    Compatible with `confuse.ConfigView`.

    :param experiment_name: Name of the experiment.
    :param experiment_path: Path where logs and artifacts are stored.
    :return: A list of PyTorch Lightning-compatible loggers.
    """
    loggers = []
    ml_loggers_cfg: MLLoggersConfig = CONFIG.logging.ml_loggers

    # TensorBoard Logger
    if ml_loggers_cfg.tensorboard.use:
        tb_log_dir = experiment_path / ml_loggers_cfg.tensorboard.save_dir
        loggers.append(TensorBoardLogger(save_dir=str(tb_log_dir), name=run_name))
    # MLflow Logger
    if ml_loggers_cfg.mlflow.use:
        ml_log_dir = experiment_path / ml_loggers_cfg.mlflow.save_dir
        loggers.append(
            MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=ml_loggers_cfg.mlflow.tracking_uri,
                run_name=run_name,
                save_dir=str(ml_log_dir),
            )
        )
    # W&B Logger
    if ml_loggers_cfg.wandb.use:
        wandb_mode = ml_loggers_cfg.wandb.mode
        os.environ["WANDB_MODE"] = wandb_mode
        wandb_log_dir = experiment_path / ml_loggers_cfg.wandb.save_dir

        loggers.append(
            WandbLogger(
                project=experiment_name,
                name=run_name,
                log_model=ml_loggers_cfg.wandb.log_model,
                save_dir=str(wandb_log_dir),
            )
        )
    if len(loggers) > 0:
        logger.info(f"Initialized ML loggers: {', '.join([type(log).__name__ for log in loggers])}")
    else:
        logger.warning("There are no enabled ML loggers. Experiments are untracked.")
    return loggers


def create_experiments_dirs(
    experiment_path: Path, model_name: str = "all_models", run_name="run_default"
) -> Dict[str, Path]:
    """
    Creates necessary directories for the experiment:
    - TensorBoard logs
    - MLflow artifacts
    - W&B logs
    - Weights for checkpoints

    :param experiment_path: Base directory of the current experiment.
    """
    path_dict = {}
    ml_loggers_cfg: MLLoggersConfig = CONFIG.logging.ml_loggers

    if ml_loggers_cfg.tensorboard.use:
        path_dict["tb_log_dir"] = experiment_path / ml_loggers_cfg.tensorboard.save_dir
        path_dict["tb_log_dir"].mkdir(parents=True, exist_ok=True)

    if ml_loggers_cfg.mlflow.use:
        path_dict["ml_log_dir"] = experiment_path / ml_loggers_cfg.mlflow.save_dir
        path_dict["ml_log_dir"].mkdir(parents=True, exist_ok=True)

    if ml_loggers_cfg.wandb.use:
        path_dict["wandb_log_dir"] = experiment_path / ml_loggers_cfg.wandb.save_dir
        path_dict["wandb_log_dir"].mkdir(parents=True, exist_ok=True)

    path_dict["weights_path"] = experiment_path / CONFIG.trainer.weights_folder / model_name / run_name
    path_dict["weights_path"].mkdir(parents=True, exist_ok=True)
    logger.info(f"Created experiment dirs in {experiment_path}.")

    return path_dict


def save_config_to_experiment(experiment_path: Path) -> None:
    """
    Copies the config.yaml file into the experiment folder for reproducibility.

    :param experiment_path: Path to the experiment base folder.
    :param config_path: Path to the source configs/env folder.
    """
    destination_path = experiment_path / "configs"

    # Copy only if not already exists
    if not destination_path.exists():
        shutil.copytree(str(CONFIG_DIR), str(destination_path))

    logger.debug(f"The ML config.yaml was successfully saved to the experiment folder: {experiment_path}.")


def log_image_to_all_loggers(ml_loggers: list, tag: str, image_tensor: torch.Tensor, epoch: int, step: int) -> None:
    """
    Logs an image to all available loggers depending on their type.
    Supports TensorBoardLogger and WandbLogger.
    Skips MLflowLogger and other unsupported loggers.

    :param loggers: list of loggers.
    :param image_tensor: A torch.Tensor image (C, H, W).
    :param tag: Name/tag for the image.
    :param epoch: epoch number
    :param step: global step number.
    """
    pil_image = compress_image_for_logging(image_tensor)

    for ml_logger in ml_loggers:
        if isinstance(ml_logger, TensorBoardLogger):
            try:
                ml_logger.experiment.add_image(tag, F.to_tensor(pil_image), global_step=epoch)
            except Exception:
                error_message = traceback.format_exc()
                logger.exception("Error with logging the image to TensorBoard:\n", error_message)
        elif isinstance(ml_logger, WandbLogger):
            try:
                ml_logger.experiment.log({tag: wandb.Image(pil_image), "epoch": epoch}, step=step)
            except Exception:
                error_message = traceback.format_exc()
                logger.exception("Error with logging the image to Weight and Biases:\n", error_message)
        elif isinstance(ml_logger, MLFlowLogger):
            # log_image_to_mlflow(logger, image_tensor, tag, step)
            try:
                ml_logger.experiment.log_image(run_id=ml_logger.run_id, image=pil_image, key=tag, step=epoch)
            except Exception:
                error_message = traceback.format_exc()
                logger.exception("Error with logging the image to MLflow:\n", error_message)


def compress_image_for_logging(image_tensor: torch.Tensor, quality: int = 75) -> Image:
    """
    Compresses the given tensor image and returns PIL Image.
    :param image_tensor: image tensor.
    :param quality: jpeg image compresssion quality
    :return: A compressed copy of the image.
    """
    image = F.to_pil_image(image_tensor)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", optimize=True, quality=quality)
    buffer.seek(0)

    return Image.open(buffer)


def log_model_weights(ml_loggers: list, model_name: str, model_weights_path: str) -> None:
    """
    Logs model weights to Weights & Biases and/or MLflow based on configuration.

    :param loggers: List of active ML loggers
    :param model: The trained model
    :param model_weights_path: Path to the saved model weights
    :param model_name: Name of the model
    """
    for ml_logger in ml_loggers:
        if isinstance(ml_logger, WandbLogger):
            try:
                # Log the model to wandb with production tag
                artifact = wandb.Artifact(
                    name=f"{model_name}-weights",
                    type="model",
                    description=f"Model weights for {model_name}",
                    metadata={"model_type": model_name},
                )
                artifact.add_file(model_weights_path)
                ml_logger.experiment.log_artifact(artifact, aliases=["latest", "production"])
                logger.info("Logged model weights to Weights & Biases with production tag")
            except Exception as e:
                logger.error(f"Failed to log model to Weights & Biases: {str(e)}")

        elif isinstance(ml_logger, MLFlowLogger):
            try:
                # Log the model to MLflow
                ml_logger.experiment.log_artifact(
                    run_id=ml_logger.run_id, local_path=model_weights_path, artifact_path="model_weights"
                )
                # Add production tag
                ml_logger.experiment.set_tag(ml_logger.run_id, "production", "true")
                logger.info("Logged model weights to MLflow with production tag")
            except Exception as e:
                logger.error(f"Failed to log model to MLflow: {str(e)}")
