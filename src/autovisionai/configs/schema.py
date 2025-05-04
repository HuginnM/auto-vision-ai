# src/autovisionai/configs/schema.py
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    data_root: Path
    test_data_root: Path


class MLLoggingConfig(BaseModel):
    root_dir: Path
    mlflow: Optional[dict] = {}
    wandb: Optional[dict] = {}
    tensorboard: Optional[dict] = {}


class TrainerConfig(BaseModel):
    weights_folder: str


class AppConfig(BaseModel):
    dataset: DatasetConfig
    ml_logging: MLLoggingConfig
    trainer: TrainerConfig
