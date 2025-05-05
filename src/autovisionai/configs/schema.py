from typing import Literal

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    data_root: str
    images_folder: str
    masks_folder: str
    test_data_root: str
    allowed_extensions: list[str]


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


class TrainerConfig(BaseModel):
    max_epoch: int
    log_every_n_steps: int
    weights_folder: str


class TensorBoardConfig(BaseModel):
    use: bool
    save_dir: str


class MLFlowConfig(BaseModel):
    use: bool
    tracking_uri: str
    save_dir: str


class WandBConfig(BaseModel):
    use: bool
    log_model: bool
    save_dir: str
    mode: Literal["offline", "online"]


class MLLoggersConfig(BaseModel):
    tensorboard: TensorBoardConfig
    mlflow: MLFlowConfig
    wandb: WandBConfig


class LoggingConfig(BaseModel):
    root_dir: str
    ml_loggers: MLLoggersConfig


class AppConfig(BaseModel):
    dataset: DatasetConfig
    data_augmentation: DataAugmentationConfig
    dataloader: DataloaderConfig
    datamodule: DataModuleConfig
    unet: UNetConfig
    fast_scnn: FastSCNNConfig
    mask_rcnn: MaskRCNNConfig
    trainer: TrainerConfig
    logging: LoggingConfig
