from pathlib import Path

import pytest
from pydantic import ValidationError

from autovisionai.core.configs.schema import (
    GlobalConfig,
)


@pytest.fixture
def minimal_valid_config_dict():
    return {
        "app": {
            "api_base_url": "http://localhost:8000",
            "ui_base_url": "http://localhost:8501",
        },
        "dataset": {
            "data_root": "./data",
            "test_data_root": "./tests/test_data",
            "images_folder": "images",
            "masks_folder": "masks",
            "allowed_extensions": [".jpg", ".png"],
        },
        "data_augmentation": {
            "bbox_min_size": 20,
            "h_flip_prob": 0.5,
            "random_crop_prob": 0.5,
            "resize_to": 512,
            "random_crop_crop_to": 512,
        },
        "dataloader": {"num_workers": 4},
        "datamodule": {"training_set_size": 0.8},
        "models": {
            "available": ["unet", "fast_scnn", "mask_rcnn"],
            "unet": {
                "in_channels": 3,
                "n_classes": 1,
                "optimizer": {"initial_lr": 0.01, "weight_decay": 0.0005},
                "lr_scheduler": {"step_size": 2, "gamma": 0.1},
            },
            "fast_scnn": {
                "n_classes": 1,
                "optimizer": {"initial_lr": 0.01, "weight_decay": 0.0005},
                "lr_scheduler": {"step_size": 3, "gamma": 0.5},
            },
            "mask_rcnn": {
                "n_classes": 2,
                "hidden_size": 256,
                "optimizer": {"initial_lr": 0.005, "momentum": 0.9, "weight_decay": 0.0005},
                "lr_scheduler": {"step_size": 1, "gamma": 0.1},
            },
        },
        "trainer": {"max_epoch": 5, "log_every_n_steps": 10, "weights_folder": "weights"},
        "logging": {
            "app_logger": {
                "stdout": {"level": "INFO", "format": "{time} - {level} - {message}"},
                "file": {
                    "level": "DEBUG",
                    "save_dir": "logs",
                    "file_name": "app.log",
                    "format": "{time} - {level} - {message}",
                    "rotation": "10 MB",
                    "backup_count": 2,
                },
            },
            "ml_loggers": {
                "root_dir": "./experiments",
                "tensorboard": {
                    "use": True,
                    "tracking_uri": "http://localhost:6006",
                },
                "mlflow": {
                    "use": False,
                    "tracking_uri": "http://localhost:5000",
                },
                "wandb": {
                    "use": False,
                    "log_model": False,
                    "tracking_uri": "https://api.wandb.ai",
                    "inference_project": "autovisionai-inference",
                    "mode": "offline",
                },
            },
        },
    }


def test_app_config_valid(minimal_valid_config_dict):
    config = GlobalConfig(**minimal_valid_config_dict)
    assert isinstance(config.dataset.data_root, Path)
    assert config.models.unet.optimizer.initial_lr == 0.01
    assert config.logging.ml_loggers.tensorboard.use is True


def test_app_config_missing_required_field(minimal_valid_config_dict):
    del minimal_valid_config_dict["dataset"]["data_root"]
    with pytest.raises(ValidationError) as exc_info:
        GlobalConfig(**minimal_valid_config_dict)
    assert "data_root" in str(exc_info.value)


def test_app_config_invalid_type(minimal_valid_config_dict):
    minimal_valid_config_dict["dataloader"]["num_workers"] = "many"
    with pytest.raises(ValidationError) as exc_info:
        GlobalConfig(**minimal_valid_config_dict)
    assert "num_workers" in str(exc_info.value)


def test_app_logger_defaults(minimal_valid_config_dict):
    config = GlobalConfig(**minimal_valid_config_dict)
    stdout = config.logging.app_logger.stdout
    assert stdout.level == "INFO"

    file = config.logging.app_logger.file
    assert file.level == "DEBUG"
    assert file.rotation == "10 MB"
    assert file.backup_count == 2
    assert file.file_name == "app.log"
