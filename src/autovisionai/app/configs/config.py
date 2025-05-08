import os
import tomllib
from pathlib import Path
from typing import List, Tuple

import yaml
from dotenv import load_dotenv

from autovisionai.app.configs.schema import AppConfig
from autovisionai.app.utils.common import find_project_root

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "local")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

PROJECT_ROOT: Path = find_project_root()
CONFIG_DIR: Path = PROJECT_ROOT / "src" / "autovisionai" / "configs" / ENV_MODE
CONFIG_FILES: List = ["data.yaml", "models.yaml", "logging.yaml"]


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
    merged["logging"]["ml_loggers"]["root_dir"] = PROJECT_ROOT / "experiments"

    return AppConfig(**merged)


def read_project_meta() -> Tuple[str, str]:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    project = data.get("project") or {}
    return project.get("name", "unknown"), project.get("version", "0.0.0")


PROJECT_NAME, PROJECT_VERSION = read_project_meta()
CONFIG: AppConfig = load_app_config()
