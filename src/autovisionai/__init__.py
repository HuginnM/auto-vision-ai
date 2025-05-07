from autovisionai.configs import CONFIG
from autovisionai.loggers import setup_app_logger

__version__ = "0.1.0"

setup_app_logger(CONFIG.logging.app_logger)
