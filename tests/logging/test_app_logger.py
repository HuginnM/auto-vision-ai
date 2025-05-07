import logging
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from autovisionai.configs import PROJECT_NAME
from autovisionai.loggers.app_logger import AppLogger


@pytest.fixture(autouse=True)
def reset_autovisionai_logger():
    """
    Automatically resets the 'autovisionai' logger before each test
    in this test module only.
    """
    logger = logging.getLogger("autovisionai")
    logger.handlers.clear()
    logger.propagate = True

    # Remove child loggers like 'autovisionai.file', 'autovisionai.submodule.x'
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("autovisionai"):
            del logging.Logger.manager.loggerDict[name]

    # Optional: clear root logger if polluted (e.g. when AppLogger used root previously)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


@pytest.fixture
def mock_config(tmp_path):
    return SimpleNamespace(
        stdout=SimpleNamespace(level="INFO", format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"),
        file=SimpleNamespace(
            level="DEBUG",
            save_dir=tmp_path / "logs",
            file_name="test.log",
            format="%(asctime)s | %(levelname)s | %(message)s",
            rotation="1 MB",
            backup_count=2,
            encoding="utf-8",
        ),
    )


@pytest.fixture(autouse=True)
def cleanup_logger_and_logs(mock_config):
    """
    Automatically runs after each test to close autovisionai logger handlers
    and optionally delete log files.
    """
    yield  # run the test
    # Clean up autovisionai logger handlers
    project_logger = logging.getLogger(PROJECT_NAME)
    for handler in project_logger.handlers:
        handler.close()
    project_logger.handlers.clear()

    # Attempt to remove test log file if it exists
    for handler in logging.Logger.manager.loggerDict.copy():
        if handler.startswith(PROJECT_NAME):
            del logging.Logger.manager.loggerDict[handler]

    temp_log_dir = mock_config.file.save_dir
    log_file = temp_log_dir / "test.log"

    if log_file.exists():
        try:
            log_file.unlink()
        except Exception as e:
            print(f"Warning: could not delete log file: {e}")

    # Optional: remove the folder if empty
    if temp_log_dir.exists():
        try:
            temp_log_dir.rmdir()
        except OSError:
            # Directory not empty
            print("Can't delete temp_log_dir - directory is not empty.")
            pass


def test_app_logger_initializes(mock_config):
    AppLogger(mock_config)
    handlers = logging.getLogger(PROJECT_NAME).handlers
    assert len(handlers) == 2
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    assert any(isinstance(h, logging.FileHandler) for h in handlers)


def test_logger_idempotent(mock_config):
    AppLogger(mock_config)
    count_1 = len(logging.getLogger(PROJECT_NAME).handlers)
    AppLogger(mock_config)
    count_2 = len(logging.getLogger(PROJECT_NAME).handlers)
    assert count_1 == count_2


def test_logger_outputs_to_stream():
    stream = StringIO()
    logger = logging.getLogger("autovisionai.stream")
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
    logger = logging.getLogger("autovisionai.file")

    test_msg = "File logging works"
    logger.debug(test_msg)

    save_dir: Path = mock_config.file.save_dir
    log_path: Path = save_dir / mock_config.file.file_name

    assert log_path.exists(), f"Expected log file at {log_path} to exist"

    content = log_path.read_text(encoding="utf-8")
    assert test_msg in content
