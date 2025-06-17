from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autovisionai.api.schemas.train import TrainingRequest
from autovisionai.api.services.train_service import TrainingProgress, TrainingService
from autovisionai.core.models.unet.unet_trainer import UnetTrainer


@pytest.fixture
def training_service():
    """Create a fresh TrainingService instance for each test."""
    return TrainingService()


@pytest.fixture
def mock_model():
    """Create a mock model with required attributes."""
    model = MagicMock()
    model.model_name = "unet"
    return model


@pytest.fixture
def mock_trainer(mock_model):
    """Create a mock trainer with all required attributes and methods."""
    trainer = MagicMock()

    # Basic attributes
    trainer.model = mock_model
    trainer.model_name = "unet"
    trainer.experiment_path = "test/path"
    trainer.current_epoch = 0
    trainer.callback_metrics = {"train/loss_epoch": 0.5}

    # Configuration
    trainer.config = MagicMock()
    trainer.config.max_epochs = 10
    trainer.config.batch_size = 4

    # Paths and logging
    trainer.exp_paths = {"weights_path": "test/path"}
    trainer.ml_loggers = []

    # Methods
    trainer.train = MagicMock()
    trainer._save_model_weights = MagicMock(return_value="test/weights.pt")
    trainer._setup_training_environment = MagicMock()
    trainer._setup_experiment_directories = MagicMock()
    trainer._setup_loggers = MagicMock()
    trainer._setup_callbacks = MagicMock()
    trainer._create_datamodule = MagicMock()

    # Callbacks
    mock_callback = MagicMock()
    mock_callback.on_epoch_end = MagicMock()
    trainer.callbacks = []  # Start with empty list, callbacks will be added

    return trainer


@pytest.fixture
def training_request():
    """Create a standard training request for testing."""
    return TrainingRequest(
        experiment_name="test_experiment",
        model_name="unet",
        batch_size=4,
        max_epochs=10,
        epoch_patience=2,
        use_resize=False,
        use_random_crop=False,
        use_hflip=False,
    )


class TestTrainingProgress:
    """Test the TrainingProgress class functionality."""

    def test_initialization(self):
        """Test that TrainingProgress initializes with correct default values."""
        progress = TrainingProgress()

        assert progress.current_epoch == -1
        assert progress.total_epochs == 0
        assert progress.current_loss == float("inf")
        assert progress.best_loss == float("inf")
        assert progress.status == "initializing"
        assert progress.detail == "Training is starting..."

    def test_update_progress(self):
        """Test updating progress values."""
        progress = TrainingProgress()

        progress.current_epoch = 5
        progress.total_epochs = 10
        progress.current_loss = 0.5
        progress.best_loss = 0.4
        progress.status = "training"
        progress.detail = "Training in progress"

        assert progress.current_epoch == 5
        assert progress.total_epochs == 10
        assert progress.current_loss == 0.5
        assert progress.best_loss == 0.4
        assert progress.status == "training"
        assert progress.detail == "Training in progress"


class TestTrainingService:
    """Test the TrainingService class functionality."""

    def test_get_model_trainer_valid_model(self, training_service):
        """Test getting a valid model trainer."""
        model = training_service.get_model_trainer("unet")
        assert isinstance(model, UnetTrainer)

    def test_get_model_trainer_invalid_model(self, training_service):
        """Test getting an invalid model trainer raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            training_service.get_model_trainer("invalid_model")

    @pytest.mark.asyncio
    async def test_train_model_success(self, training_service, training_request, mock_trainer):
        """Test successful model training."""
        with (
            patch("autovisionai.api.services.train_service.UnetTrainer", return_value=mock_trainer.model),
            patch("autovisionai.api.services.train_service.ModelTrainer", return_value=mock_trainer),
            patch("asyncio.get_running_loop"),
        ):
            progress_callback = AsyncMock()
            result = await training_service.train_model(training_request, progress_callback)

            # Verify success response
            assert result["status"] == "success"
            assert result["detail"] == "Training completed successfully"
            assert result["experiment_path"] == "test/path"
            assert result["model_weights_path"] == "test/weights.pt"

            # Verify trainer.train was called
            mock_trainer.train.assert_called_once()

            # Verify progress tracking was set up and cleaned up
            assert training_request.experiment_name not in training_service.active_trainings

    @pytest.mark.asyncio
    async def test_train_model_error(self, training_service, training_request):
        """Test model training with error handling."""
        with patch("autovisionai.api.services.train_service.UnetTrainer", side_effect=Exception("Test error")):
            progress_callback = AsyncMock()
            result = await training_service.train_model(training_request, progress_callback)

            # Verify error response
            assert result["status"] == "error"
            assert "Test error" in result["detail"]
            assert result["experiment_path"] is None
            assert result["model_weights_path"] is None

            # Verify training was removed from active trainings
            assert training_request.experiment_name not in training_service.active_trainings

    def test_get_training_progress_existing(self, training_service):
        """Test getting progress for an existing training job."""
        # Set up test data
        progress = TrainingProgress()
        progress.current_epoch = 5
        progress.total_epochs = 10
        progress.current_loss = 0.5
        progress.best_loss = 0.4
        progress.status = "training"
        progress.detail = "Training in progress"

        training_service.active_trainings["test_experiment"] = progress

        # Get progress
        retrieved_progress = training_service.get_training_progress("test_experiment")

        # Verify progress data
        assert retrieved_progress is not None
        assert retrieved_progress.current_epoch == 5
        assert retrieved_progress.total_epochs == 10
        assert retrieved_progress.current_loss == 0.5
        assert retrieved_progress.best_loss == 0.4
        assert retrieved_progress.status == "training"
        assert retrieved_progress.detail == "Training in progress"

    def test_get_training_progress_nonexistent(self, training_service):
        """Test getting progress for a nonexistent training job."""
        progress = training_service.get_training_progress("nonexistent")

        # Should return None for nonexistent training
        assert progress is None
