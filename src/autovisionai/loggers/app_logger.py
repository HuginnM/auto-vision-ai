import sys

from loguru import logger

from autovisionai.configs import CONFIG
from autovisionai.utils.pathing import find_project_root


def setup_logger() -> None:
    stdout_cfg = CONFIG.logging.global_logger.stdout
    file_cfg = CONFIG.logging.global_logger.file

    log_file_path = find_project_root() / file_cfg.save_dir / file_cfg.file_name
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stdout,
        level=stdout_cfg.level,
        format=stdout_cfg.format,
        backtrace=stdout_cfg.backtrace,
        diagnose=stdout_cfg.diagnose,
        enqueue=stdout_cfg.enqueue,
    )

    logger.add(
        str(log_file_path),
        level=file_cfg.level,
        format=file_cfg.format,
        rotation=file_cfg.rotation,
        retention=file_cfg.retention,
        encoding=file_cfg.encoding,
        backtrace=file_cfg.backtrace,
        diagnose=file_cfg.diagnose,
        enqueue=file_cfg.enqueue,
    )
