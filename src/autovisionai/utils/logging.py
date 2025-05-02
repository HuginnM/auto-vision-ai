import datetime as dt
import io
import os
import shutil
import traceback
from pathlib import Path

import torch
import torchvision.transforms.functional as F
import wandb
from PIL import Image
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger

# from pytorch_lightning.loggers.logger import LoggerCollection
from autovisionai.configs.config import CONFIG, config_file_path


def get_run_name():
    local_tz = dt.datetime.now().astimezone().tzinfo
    return f"run_{dt.datetime.now(tz=local_tz).strftime('%Y%m%dT%H%M%SUTC%z')}".replace("+", "")  # ISO8601 format


def get_loggers(experiment_name: str, experiment_path: Path) -> list:
    """
    Creates a list of loggers based on the provided config.
    Supports TensorBoard, MLflow, and Weights & Biases (W&B).
    Compatible with `confuse.ConfigView`.

    :param experiment_name: Name of the experiment.
    :param experiment_path: Path where logs and artifacts are stored.
    :return: A list of PyTorch Lightning-compatible loggers.
    """
    loggers = []
    run_name = get_run_name()

    # TensorBoard Logger
    if CONFIG["logging"]["tensorboard"]["use"].get(bool):
        tb_log_dir = experiment_path / CONFIG["logging"]["tensorboard"]["save_dir"].get(str)
        loggers.append(TensorBoardLogger(save_dir=str(tb_log_dir), name=run_name))

    # MLflow Logger
    if CONFIG["logging"]["mlflow"]["use"].get(bool):
        mlflow_uri = CONFIG["logging"]["mlflow"]["tracking_uri"].get(str)
        ml_log_dir = experiment_path / CONFIG["logging"]["mlflow"]["save_dir"].get(str)
        loggers.append(
            MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=mlflow_uri,
                run_name=run_name,
                save_dir=str(ml_log_dir),
            )
        )

    # W&B Logger
    if CONFIG["logging"]["wandb"]["use"].get(bool):
        wandb_mode = CONFIG["logging"]["wandb"]["mode"].get(str)
        os.environ["WANDB_MODE"] = wandb_mode
        wandb_log_dir = experiment_path / CONFIG["logging"]["wandb"]["save_dir"].get(str)

        loggers.append(
            WandbLogger(
                project=experiment_name,
                name=run_name,
                log_model=CONFIG["logging"]["wandb"]["log_model"].get(bool),
                save_dir=str(wandb_log_dir),
            )
        )

    return loggers


def create_experiments_dirs(experiment_path: Path) -> None:
    """
    Creates necessary directories for the experiment:
    - TensorBoard logs
    - MLflow artifacts
    - W&B logs
    - Weights for checkpoints

    :param experiment_path: Base directory of the current experiment.
    """
    path_dict = {}

    if CONFIG["logging"]["tensorboard"]["use"].get(bool):
        path_dict["tb_log_dir"] = experiment_path / CONFIG["logging"]["tensorboard"]["save_dir"].get(str)
        path_dict["tb_log_dir"].mkdir(parents=True, exist_ok=True)

    if CONFIG["logging"]["mlflow"]["use"].get(bool):
        path_dict["ml_log_dir"] = experiment_path / CONFIG["logging"]["mlflow"]["save_dir"].get(str)
        path_dict["ml_log_dir"].mkdir(parents=True, exist_ok=True)

    if CONFIG["logging"]["wandb"]["use"].get(bool):
        path_dict["wandb_log_dir"] = experiment_path / CONFIG["logging"]["wandb"]["save_dir"].get(str)
        path_dict["wandb_log_dir"].mkdir(parents=True, exist_ok=True)

    path_dict["weights_path"] = experiment_path / CONFIG["trainer"]["weights_folder"].get(str)
    path_dict["weights_path"].mkdir(parents=True, exist_ok=True)

    return path_dict


def save_config_to_experiment(experiment_path: Path) -> None:
    """
    Copies the config.yaml file into the experiment folder for reproducibility.

    :param experiment_path: Path to the experiment base folder.
    :param config_path: Path to the source config.yaml file.
    """
    destination_path = experiment_path / "config.yaml"

    # Copy only if not already exists
    if not destination_path.exists():
        shutil.copy(str(config_file_path), str(destination_path))


def log_image_to_all_loggers(loggers: list, tag: str, image_tensor: torch.Tensor, epoch: int, step: int) -> None:
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

    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            try:
                logger.experiment.add_image(tag, F.to_tensor(pil_image), global_step=epoch)
            except Exception:
                error_message = traceback.format_exc()
                print("Error with logging the image to TensorBoard:\n", error_message)
        elif isinstance(logger, WandbLogger):
            try:
                logger.experiment.log({tag: wandb.Image(pil_image), "epoch": epoch}, step=step)
            except Exception:
                error_message = traceback.format_exc()
                print("Error with logging the image to Weight and Biases:\n", error_message)
        elif isinstance(logger, MLFlowLogger):
            # log_image_to_mlflow(logger, image_tensor, tag, step)
            try:
                logger.experiment.log_image(run_id=logger.run_id, image=pil_image, key=tag, step=epoch)
            except Exception:
                error_message = traceback.format_exc()
                print("Error with logging the image to MLflow:\n", error_message)


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
