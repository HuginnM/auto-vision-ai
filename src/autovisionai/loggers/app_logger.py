import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from autovisionai.configs import AppLoggerConfig
from autovisionai.utils.common import parse_size


class AppLogger:
    def __init__(self, config: AppLoggerConfig):
        self.config = config
        self.logger = logging.getLogger()

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
        os.makedirs(self.config.file.save_dir, exist_ok=True)
        self._log_file_path = os.path.join(self.config.file.save_dir, self.config.file.file_name)

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
