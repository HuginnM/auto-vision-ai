"""Tests for the training pipeline.

This module contains tests for the ModelTrainer class and its components.
It includes unit tests, integration tests, and mock tests for various scenarios.
"""

import logging
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import Logger

from autovisionai.configs import CONFIG
from autovisionai.processing.datamodule import CarsDataModule
from autovisionai.train import ModelTrainer, TrainingConfig

# Disable ML logging for tests
os.environ["WANDB_MODE"] = "disabled"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
CONFIG.logging.ml_loggers.wandb.use = False

logger = logging.getLogger(__name__)


class MockModel(LightningModule):
    """Mock model for testing purposes."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(10, 2)
        self.model_name = "mock_model"

    def _get_name(self) -> str:
        return self.model_name

    def training_step(self, batch, batch_idx):
        """Required PyTorch Lightning method."""
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Required PyTorch Lightning method."""
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val/loss_epoch", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Required PyTorch Lightning method."""
        return torch.optim.Adam(self.parameters())


class MockDataModule(CarsDataModule):
    """Mock data module for testing purposes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create dummy data
        self.train_data = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 2))
        self.val_data = torch.utils.data.TensorDataset(torch.randn(20, 10), torch.randn(20, 2))

    def setup(self, stage=None):
        """Set up the data module."""
        pass  # No setup needed for mock data

    def train_dataloader(self):
        """Return a DataLoader for training."""
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 workers for testing
        )

    def val_dataloader(self):
        """Return a DataLoader for validation."""
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 workers for testing
        )


@pytest.fixture
def mock_model():
    """Fixture that provides a mock model for testing."""
    return MockModel()


@pytest.fixture
def mock_datamodule():
    """Fixture that provides a mock data module for testing."""
    return MockDataModule(
        data_root=CONFIG.dataset.data_root,
        batch_size=2,
        num_workers=0,  # Use 0 workers for testing
        resize=False,
        random_crop=False,
        hflip=False,
        bbox=False,
    )


@pytest.fixture
def training_config():
    """Fixture that provides a training configuration for testing."""
    return TrainingConfig(
        experiment_name="test_experiment",
        batch_size=2,
        epoch_patience=1,
        use_resize=True,
        use_random_crop=True,
        use_hflip=True,
    )


@pytest.fixture
def model_trainer(mock_model, training_config):
    """Fixture that provides a ModelTrainer instance for testing."""
    return ModelTrainer(
        experiment_name=training_config.experiment_name,
        model=mock_model,
        batch_size=training_config.batch_size,
        epoch_patience=training_config.epoch_patience,
        use_resize=training_config.use_resize,
        use_random_crop=training_config.use_random_crop,
        use_hflip=training_config.use_hflip,
    )


class TestTrainingConfig:
    """Tests for the TrainingConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TrainingConfig(experiment_name="test")
        assert config.batch_size == 4
        assert config.epoch_patience == 2
        assert not config.use_resize
        assert not config.use_random_crop
        assert not config.use_hflip
        assert config.max_epochs == CONFIG.trainer.max_epoch
        assert config.log_every_n_steps == CONFIG.trainer.log_every_n_steps

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = TrainingConfig(
            experiment_name="test",
            batch_size=8,
            epoch_patience=3,
            use_resize=True,
            use_random_crop=True,
            use_hflip=True,
        )
        assert config.batch_size == 8
        assert config.epoch_patience == 3
        assert config.use_resize
        assert config.use_random_crop
        assert config.use_hflip


class TestModelTrainer:
    """Tests for the ModelTrainer class."""

    def test_initialization(self, model_trainer, training_config):
        """Test that ModelTrainer initializes correctly."""
        assert model_trainer.config == training_config
        assert model_trainer.device in ["gpu", "cpu"]
        assert isinstance(model_trainer.model, LightningModule)
        assert model_trainer.run_name is not None
        assert model_trainer.model_name == "mock_model"

    def test_max_epochs_override(self, mock_model):
        """Test that max_epochs parameter properly overrides the config default."""
        # First ensure config has a different value
        original_max_epoch = CONFIG.trainer.max_epoch
        CONFIG.trainer.max_epoch = 5  # Set to a different value

        try:
            custom_epochs = 10
            trainer = ModelTrainer(
                experiment_name="test",
                model=mock_model,
                max_epochs=custom_epochs,
            )
            assert trainer.config.max_epochs == custom_epochs
            assert trainer.config.max_epochs != CONFIG.trainer.max_epoch
        finally:
            # Restore original config value
            CONFIG.trainer.max_epoch = original_max_epoch

    def test_setup_experiment_directories(self, model_trainer):
        """Test that experiment directories are created correctly."""
        model_trainer._setup_experiment_directories()
        assert model_trainer.experiment_path.exists()
        assert model_trainer.exp_paths["weights_path"].exists()
        assert (model_trainer.experiment_path / "configs").exists()

    def test_setup_callbacks(self, model_trainer):
        """Test that callbacks are set up correctly."""
        model_trainer._setup_callbacks()
        assert len(model_trainer.callbacks) == 2
        assert any(isinstance(cb, ModelCheckpoint) for cb in model_trainer.callbacks)
        assert any(isinstance(cb, EarlyStopping) for cb in model_trainer.callbacks)

    @patch("autovisionai.train.get_loggers")
    def test_setup_loggers(self, mock_get_loggers, model_trainer):
        """Test that loggers are set up correctly."""
        mock_loggers = [MagicMock(spec=Logger)]
        mock_get_loggers.return_value = mock_loggers
        model_trainer._setup_loggers()
        assert model_trainer.ml_loggers == mock_loggers
        mock_get_loggers.assert_called_once()

    @patch("autovisionai.train.CarsDataModule")
    def test_create_datamodule(self, mock_datamodule, model_trainer):
        """Test that data module is created correctly."""
        mock_instance = MagicMock()
        mock_datamodule.return_value = mock_instance
        datamodule = model_trainer._create_datamodule()
        assert datamodule == mock_instance
        mock_datamodule.assert_called_once()

    def test_save_model_weights(self, model_trainer):
        """Test that model weights are saved correctly."""
        model_trainer._setup_experiment_directories()
        weights_path = model_trainer._save_model_weights()
        assert weights_path.exists()
        assert weights_path.suffix == ".pt"

    @patch("autovisionai.train.pl.Trainer")
    @patch("autovisionai.train.CarsDataModule")
    def test_train_success(self, mock_datamodule, mock_trainer, model_trainer):
        """Test successful training execution."""
        # Setup mocks
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_datamodule_instance = MagicMock()
        mock_datamodule.return_value = mock_datamodule_instance

        # Execute training
        model_trainer.train()

        # Verify trainer was called correctly
        mock_trainer.assert_called_once()
        mock_trainer_instance.fit.assert_called_once_with(model_trainer.model, mock_datamodule_instance)

    @patch("autovisionai.train.pl.Trainer")
    @patch("autovisionai.train.CarsDataModule")
    def test_train_failure(self, mock_datamodule, mock_trainer, model_trainer):
        """Test training failure handling."""
        # Setup mocks to raise an exception
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.fit.side_effect = Exception("Training failed")
        mock_trainer.return_value = mock_trainer_instance

        # Verify that the exception is properly wrapped
        with pytest.raises(RuntimeError) as exc_info:
            model_trainer.train()
        assert "Training failed" in str(exc_info.value)


@pytest.mark.integration
class TestModelTrainerIntegration:
    """Integration tests for the ModelTrainer class."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down test environment."""
        # Create a temporary directory for test artifacts
        self.test_dir = Path("test_artifacts")
        self.test_dir.mkdir(exist_ok=True)
        yield
        # Clean up after tests
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                # If we can't delete the directory, try to delete its contents
                for item in self.test_dir.glob("**/*"):
                    try:
                        if item.is_file():
                            item.unlink()
                    except PermissionError:
                        logger.warning(f"Could not delete file: {item}")

    def test_full_training_cycle(self, mock_model, mock_datamodule):
        """Test a complete training cycle with a mock model."""
        # Configure test environment
        CONFIG.logging.ml_loggers.root_dir = self.test_dir

        # Create and run trainer
        trainer = ModelTrainer(
            experiment_name="integration_test",
            model=mock_model,
            batch_size=2,
            max_epochs=1,  # Minimal training for testing
            datamodule=mock_datamodule,  # Use mock datamodule
        )

        # Execute training
        trainer.train()

        # Verify results
        assert (trainer.exp_paths["weights_path"] / "model.pt").exists()
        assert (trainer.experiment_path / "configs").exists()
