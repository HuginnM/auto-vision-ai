import logging
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from autovisionai.loggers.app_logger import AppLogger


@pytest.fixture(autouse=True)
def clean_logger():
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.propagate = True
    yield
    logger.handlers.clear()
    logger.propagate = True


@pytest.fixture
def mock_config(tmp_path):
    return SimpleNamespace(
        stdout=SimpleNamespace(level="INFO", format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"),
        file=SimpleNamespace(
            level="DEBUG",
            save_dir=str(tmp_path / "logs"),
            file_name="test.log",
            format="%(asctime)s | %(levelname)s | %(message)s",
            rotation="1 MB",
            backup_count=2,
            encoding="utf-8",
        ),
    )


def test_app_logger_initializes(mock_config):
    AppLogger(mock_config)
    handlers = logging.getLogger().handlers
    assert len(handlers) == 2
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    assert any(isinstance(h, logging.FileHandler) for h in handlers)


def test_logger_idempotent(mock_config):
    AppLogger(mock_config)
    count_1 = len(logging.getLogger().handlers)
    AppLogger(mock_config)
    count_2 = len(logging.getLogger().handlers)
    assert count_1 == count_2


def test_logger_outputs_to_stream():
    stream = StringIO()
    logger = logging.getLogger("test.stream")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.info("Test stream log")

    stream.seek(0)
    output = stream.read()
    assert "INFO - Test stream log" in output


def test_file_log_written(mock_config):
    AppLogger(mock_config)
    logger = logging.getLogger("test.file")

    test_msg = "File logging works"
    logger.debug(test_msg)

    log_path = Path(mock_config.file.save_dir) / mock_config.file.file_name
    assert log_path.exists()

    content = log_path.read_text(encoding="utf-8")
    assert test_msg in content

    # Clean up
    log_path.unlink()
