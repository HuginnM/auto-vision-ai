import sys
from pathlib import Path

from loguru import logger

LOG_FILE = Path(__file__).resolve().parents[3] / "logs" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}"
    "</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

logger.add(
    str(LOG_FILE),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="10 days",
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)
