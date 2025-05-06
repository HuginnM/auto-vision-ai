from pathlib import Path


def find_project_root(anchor_filename="pyproject.toml") -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / anchor_filename).exists():
            return parent
    raise FileNotFoundError(f"Could not find {anchor_filename} in parent folders.")
