from pathlib import Path

import confuse
import yaml


def find_project_root(anchor_filename="pyproject.toml") -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / anchor_filename).exists():
            return parent
    raise FileNotFoundError(f"Could not find {anchor_filename} in parent folders.")


# Get root
project_root = find_project_root()

config_file_path = project_root / "src" / "autovisionai" / "configs" / "config.yaml"
config_folder_path = project_root / "src" / "autovisionai" / "configs"
data_folder_path = project_root / "data"
test_data_folder_path = project_root / "tests" / "test_data"
experiments_folder_path = project_root / "experiments"

with open(config_file_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config["dataset"]["data_root"] = str(data_folder_path)
config["dataset"]["test_data_root"] = str(test_data_folder_path)
config["trainer"]["logs_and_weights_root"] = str(experiments_folder_path)

with open(config_file_path, "w") as f:
    yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)

CONFIG = confuse.Configuration(str(config_folder_path))
