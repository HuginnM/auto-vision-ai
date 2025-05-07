# autovisionai/loggers/app_logger.py

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from autovisionai.configs import AppLoggerConfig


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
            maxBytes=self._parse_size(self.config.file.rotation),
            backupCount=self._parse_retention(self.config.file.retention),
            encoding=self.config.file.encoding,
        )

        formatter = logging.Formatter(self.config.file.format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        handler.setLevel(self.config.file.level)
        self.logger.addHandler(handler)

    @staticmethod
    def _parse_size(size_str: str) -> int:
        """
        Parses a human-readable size string like '10 MB' or '1024 B' into bytes.
        Supports B, KB, MB, GB, TB.
        """
        size_str = size_str.strip().upper()
        try:
            num_str, unit = size_str.split()
        except ValueError as err:
            raise ValueError(f"Invalid size format: '{size_str}', expected format like '10 MB'") from err

        num = float(num_str)

        byte_factors = {
            "B": 1,
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
        }

        if unit not in byte_factors:
            raise ValueError(f"Unsupported unit '{unit}'. Supported: {', '.join(byte_factors.keys())}")

        return int(num * byte_factors[unit])


def setup_app_logger(config: AppLoggerConfig):
    """
    Public entrypoint for logger setup using AppLoggerConfig from AppConfig.
    """
    AppLogger(config)
