import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from autovisionai.core.configs import PROJECT_NAME, PROJECT_ROOT, AppLoggerConfig
from autovisionai.core.utils.common import parse_size


class AppLogger:
    def __init__(self, config: AppLoggerConfig):
        self.config = config
        self.logger = logging.getLogger(PROJECT_NAME)

        if self.logger.handlers:
            return  # Already configured

        self.logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter by own level

        self._setup_stdout_handler()
        self._setup_file_handler()
        self.logger.propagate = False

        self.logger.info(
            "AppLogger initialized", extra={"stdout_level": self.config.stdout.level, "file_path": self._log_file_path}
        )

    def _setup_stdout_handler(self):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(self.config.stdout.format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        handler.setLevel(self.config.stdout.level)
        self.logger.addHandler(handler)

    def _setup_file_handler(self):
        log_dir_path = Path(self.config.file.save_dir)

        if not log_dir_path.is_absolute():
            log_dir_path = PROJECT_ROOT / log_dir_path

        log_dir_path.mkdir(exist_ok=True, parents=True)

        self._log_file_path = log_dir_path / self.config.file.file_name

        handler = RotatingFileHandler(
            self._log_file_path,
            maxBytes=parse_size(self.config.file.rotation),
            backupCount=self.config.file.backup_count,
            encoding=self.config.file.encoding,
        )

        formatter = logging.Formatter(self.config.file.format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        handler.setLevel(self.config.file.level)
        self.logger.addHandler(handler)


def setup_app_logger(config: AppLoggerConfig):
    """
    Public entrypoint for logger setup using AppLoggerConfig from AppConfig.
    """
    AppLogger(config)
