import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from autovisionai.configs.schema import AppConfig
from autovisionai.utils.pathing import find_project_root

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "local")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

project_root = find_project_root()
config_dir = project_root / "src" / "autovisionai" / "configs" / ENV_MODE


def load_yaml_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_app_config() -> AppConfig:
    merged = {}
    for name in ["data.yaml", "models.yaml", "logging.yaml"]:
        merged.update(load_yaml_config(config_dir / name))

    # Inject runtime paths
    merged["dataset"]["data_root"] = str(project_root / "data")
    merged["dataset"]["test_data_root"] = str(project_root / "tests" / "test_data")
    merged["logging"]["root_dir"] = str(project_root / "experiments")

    return AppConfig(**merged)


CONFIG = load_app_config()
