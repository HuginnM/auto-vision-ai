from autovisionai.core.configs import CONFIG, PROJECT_VERSION
from autovisionai.core.loggers import setup_app_logger

__version__ = PROJECT_VERSION

setup_app_logger(CONFIG.logging.app_logger)
