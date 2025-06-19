import os
import tomllib
from pathlib import Path
from typing import List, Tuple

import yaml
from dotenv import load_dotenv

from autovisionai.core.configs.schema import GlobalConfig
from autovisionai.core.utils.common import find_project_root

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "local")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

PROJECT_ROOT: Path = find_project_root()
CONFIG_DIR: Path = PROJECT_ROOT / "src" / "autovisionai" / "core" / "configs" / ENV_MODE
CONFIG_FILES: List = ["app.yaml", "data.yaml", "models.yaml", "logging.yaml"]


def load_yaml_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_app_config() -> GlobalConfig:
    merged = {}
    for name in CONFIG_FILES:
        merged.update(load_yaml_config(CONFIG_DIR / name))

    # Inject runtime paths
    merged["dataset"]["data_root"] = PROJECT_ROOT / "data"
    merged["dataset"]["test_data_root"] = PROJECT_ROOT / "tests" / "test_data"
    merged["logging"]["ml_loggers"]["root_dir"] = PROJECT_ROOT / "experiments"

    return GlobalConfig(**merged)


def read_project_meta() -> Tuple[str, str]:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    project = data.get("project") or {}
    return project.get("name", "unknown"), project.get("version", "0.0.0")


PROJECT_NAME, PROJECT_VERSION = read_project_meta()
CONFIG: GlobalConfig = load_app_config()
