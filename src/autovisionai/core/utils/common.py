from pathlib import Path


def find_project_root(anchor_filename: str = "pyproject.toml") -> Path:
    """
    Walks up the directory tree to find the project root by looking for a specific anchor file.

    Args:
        anchor_filename (str): The filename to look for (e.g. "pyproject.toml").

    Returns:
        Path: The path to the project root containing the anchor file.

    Raises:
        FileNotFoundError: If the anchor file is not found in any parent directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / anchor_filename).exists():
            return parent
    raise FileNotFoundError(f"Could not find {anchor_filename} in parent folders.")


def parse_size(size_str: str) -> int:
    """
    Parses a human-readable file size string into bytes.

    Supports units: B, KB, MB, GB, TB, BIT, KBIT, MBIT, GBIT, TBIT (case-insensitive).

    Args:
        size_str (str): A string like "10 MB", "512 kbit", etc.

    Returns:
        int: Size in bytes.

    Raises:
        ValueError: If the format is invalid or unit is not supported.
    """
    size_str = size_str.strip().upper()
    try:
        num_str, unit = size_str.split()
    except ValueError as err:
        raise ValueError(f"Invalid size format: '{size_str}'") from err

    num = float(num_str)

    byte_factors = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "BIT": 1 / 8,
        "KBIT": 1024 / 8,
        "MBIT": 1024**2 / 8,
        "GBIT": 1024**3 / 8,
        "TBIT": 1024**4 / 8,
    }

    if unit not in byte_factors:
        raise ValueError(f"Unsupported unit '{unit}'")

    return int(num * byte_factors[unit])
