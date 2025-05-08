from pathlib import Path
from typing import List, Literal, Tuple

from pydantic import BaseModel, field_validator

from autovisionai.utils.common import parse_size


class DatasetConfig(BaseModel):
    data_root: Path
    images_folder: str
    masks_folder: str
    test_data_root: Path
    allowed_extensions: Tuple[str, ...]


class DataAugmentationConfig(BaseModel):
    bbox_min_size: int
    h_flip_prob: float
    random_crop_prob: float
    resize_to: int
    random_crop_crop_to: int


class DataloaderConfig(BaseModel):
    num_workers: int


class DataModuleConfig(BaseModel):
    training_set_size: float


class OptimizerConfig(BaseModel):
    initial_lr: float
    weight_decay: float | None = None
    momentum: float | None = None


class LRSchedulerConfig(BaseModel):
    step_size: int
    gamma: float


class UNetConfig(BaseModel):
    in_channels: int
    n_classes: int
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig


class FastSCNNConfig(BaseModel):
    n_classes: int
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig


class MaskRCNNConfig(BaseModel):
    n_classes: int
    hidden_size: int
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig


class MLModelsConfig(BaseModel):
    available: List
    unet: UNetConfig
    fast_scnn: FastSCNNConfig
    mask_rcnn: MaskRCNNConfig


class TrainerConfig(BaseModel):
    max_epoch: int
    log_every_n_steps: int
    weights_folder: str


class TensorBoardConfig(BaseModel):
    use: bool
    save_dir: str = "tensorlogs"


class MLFlowConfig(BaseModel):
    use: bool
    tracking_uri: str
    save_dir: str = "mlruns"


class WandBConfig(BaseModel):
    use: bool
    log_model: bool
    save_dir: str = "wandb"
    mode: Literal["offline", "online"]


class MLLoggersConfig(BaseModel):
    root_dir: Path
    tensorboard: TensorBoardConfig
    mlflow: MLFlowConfig
    wandb: WandBConfig


class StdoutLoggerConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: str


class FileLoggerConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    save_dir: str
    file_name: str
    format: str
    rotation: str
    backup_count: int
    encoding: str = "utf-8"

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, v: str) -> str:
        _ = parse_size(v)
        return v


class AppLoggerConfig(BaseModel):
    stdout: StdoutLoggerConfig
    file: FileLoggerConfig


class LoggingConfig(BaseModel):
    app_logger: AppLoggerConfig
    ml_loggers: MLLoggersConfig


class AppConfig(BaseModel):
    dataset: DatasetConfig
    data_augmentation: DataAugmentationConfig
    dataloader: DataloaderConfig
    datamodule: DataModuleConfig
    models: MLModelsConfig
    trainer: TrainerConfig
    logging: LoggingConfig
