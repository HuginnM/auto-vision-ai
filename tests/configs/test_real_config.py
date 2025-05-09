from autovisionai.core.configs.config import load_app_config
from autovisionai.core.configs.schema import AppConfig


def test_load_real_app_config():
    """Test that the real merged YAML config loads without validation errors."""
    config = load_app_config()
    assert isinstance(config, AppConfig)


def test_real_config_dataset_fields():
    config = load_app_config()
    assert config.dataset.data_root.exists()
    assert config.dataset.test_data_root.exists()
    assert config.dataset.images_folder != ""
    assert config.dataset.masks_folder != ""
    assert isinstance(config.dataset.allowed_extensions, tuple)
    assert all(ext.startswith(".") for ext in config.dataset.allowed_extensions)


def test_real_config_model_unet_optimizer():
    config = load_app_config()
    unet = config.models.unet
    assert unet.in_channels > 0
    assert unet.n_classes >= 1
    assert unet.optimizer.initial_lr > 0
    assert unet.lr_scheduler.step_size > 0
    assert 0.0 < unet.lr_scheduler.gamma <= 1.0


def test_real_logging_tensorboard_settings():
    config = load_app_config()
    tb = config.logging.ml_loggers.tensorboard
    assert isinstance(tb.use, bool)
    if tb.use:
        assert tb.save_dir != ""


def test_real_logging_file_rotation_format():
    config = load_app_config()
    file_log = config.logging.app_logger.file
    assert file_log.rotation.endswith("MB") or file_log.rotation.endswith("GB")
    assert file_log.format != ""
