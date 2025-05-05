import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from autovisionai.configs.schema import AppConfig
from autovisionai.utils.pathing import find_project_root

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "local")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

PROJECT_ROOT: Path = find_project_root()
CONFIG_DIR: Path = PROJECT_ROOT / "src" / "autovisionai" / "configs" / ENV_MODE
CONFIG_FILES: list = ["data.yaml", "models.yaml", "logging.yaml"]


def load_yaml_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_app_config() -> AppConfig:
    merged = {}
    for name in CONFIG_FILES:
        merged.update(load_yaml_config(CONFIG_DIR / name))

    # Inject runtime paths
    merged["dataset"]["data_root"] = PROJECT_ROOT / "data"
    merged["dataset"]["test_data_root"] = PROJECT_ROOT / "tests" / "test_data"
    merged["logging"]["root_dir"] = PROJECT_ROOT / "experiments"

    return AppConfig(**merged)


CONFIG = load_app_config()
