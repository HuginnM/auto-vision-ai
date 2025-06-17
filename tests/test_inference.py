"""Tests for the inference pipeline.

This module contains tests for the InferenceEngine class and its components.
It includes unit tests, integration tests, and mock tests for various scenarios.
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from autovisionai.core.configs import CONFIG
from autovisionai.core.inference import InferenceEngine, ModelRegistry

# Disable ML logging for tests
os.environ["WANDB_MODE"] = "disabled"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
CONFIG.logging.ml_loggers.wandb.use = False
CONFIG.logging.ml_loggers.mlflow.use = False

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_image_tensor():
    """Fixture that provides a mock image tensor for testing."""
    return torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image


@pytest.fixture
def mock_raw_mask():
    """Fixture that provides a mock raw mask for testing."""
    return np.random.rand(1, 1, 224, 224)  # Batch size 1, 1 channel, 224x224 mask


@pytest.fixture
def mock_artifact():
    """Fixture that provides a mock W&B artifact for testing."""
    mock_artifact = MagicMock()
    mock_artifact.version = "v1"
    mock_artifact.download.return_value = "mock_artifact_dir"
    return mock_artifact


class TestInferenceEngine:
    """Tests for the InferenceEngine class."""

    @patch("wandb.init")
    @patch("wandb.use_artifact")
    @patch("autovisionai.core.inference.unet_inference")
    def test_initialization(
        self,
        mock_unet_inference,
        mock_use_artifact,
        mock_wandb_init,
        mock_artifact,
    ):
        """Test that InferenceEngine initializes correctly."""
        # Setup mocks
        mock_wandb_init.return_value = MagicMock()
        mock_artifact.version = "v1"
        mock_use_artifact.return_value = mock_artifact
        mock_unet_inference.return_value = np.random.rand(1, 1, 224, 224)

        # Create test weights file
        weights_path = Path("mock_artifact_dir") / "model.pt"
        weights_path.parent.mkdir(exist_ok=True)
        torch.save(torch.nn.Linear(10, 2).state_dict(), weights_path)

        try:
            engine = InferenceEngine("unet")
            assert engine.model_name == "unet"
            assert engine.device.type in ["cuda", "cpu"]  # Check device type instead of device object
            assert engine.model_version is None  # Model version is not set during initialization

            # Call infer to trigger weights loading
            engine.infer(torch.randn(1, 3, 224, 224))
            assert engine.model_version == "v1"  # Model version is set after loading weights

            mock_wandb_init.assert_called_once()
            mock_use_artifact.assert_called_once()
            mock_unet_inference.assert_called_once()
        finally:
            # Cleanup
            if weights_path.exists():
                weights_path.unlink()
            if weights_path.parent.exists():
                weights_path.parent.rmdir()

    @patch("wandb.init")
    @patch("wandb.use_artifact")
    def test_load_weights_failure(self, mock_use_artifact, mock_wandb_init):
        """Test handling of weight loading failure."""
        # Setup mocks to raise an exception
        mock_wandb_init.return_value = MagicMock()
        mock_use_artifact.side_effect = Exception("Failed to load artifact")

        with pytest.raises(Exception) as exc_info:
            engine = InferenceEngine("unet")
            engine.infer(torch.randn(1, 3, 224, 224))  # Try to load weights during inference
        assert "Failed to load artifact" in str(exc_info.value)
        mock_use_artifact.assert_called_once_with("wandb-registry-model/unet:production", type="model")

    @patch("wandb.init")
    @patch("wandb.use_artifact")
    def test_process_results(self, mock_use_artifact, mock_wandb_init, mock_image_tensor, mock_raw_mask, mock_artifact):
        """Test processing of inference results."""
        # Setup mocks
        mock_wandb_init.return_value = MagicMock()
        mock_use_artifact.return_value = mock_artifact

        # Create test weights file
        weights_path = Path("mock_artifact_dir") / "model.pt"
        weights_path.parent.mkdir(exist_ok=True)
        torch.save(torch.nn.Linear(10, 2).state_dict(), weights_path)

        try:
            engine = InferenceEngine("unet")

            # Process results
            processed_image, processed_mask, binary_mask = engine.process_results(
                mock_image_tensor, mock_raw_mask, threshold=0.5
            )

            # Verify shapes and types
            assert isinstance(processed_image, np.ndarray)
            assert isinstance(processed_mask, np.ndarray)
            assert isinstance(binary_mask, np.ndarray)
            assert processed_image.shape == (224, 224, 3)
            assert processed_mask.shape == (224, 224, 1)
            assert binary_mask.shape == (224, 224, 1)
            assert binary_mask.dtype == np.uint8
            assert np.all(np.unique(binary_mask) == np.array([0, 255]))
        finally:
            # Cleanup
            if weights_path.exists():
                weights_path.unlink()
            if weights_path.parent.exists():
                weights_path.parent.rmdir()

    @patch("autovisionai.core.inference.unet_inference")
    @patch("autovisionai.core.inference.log_inference_results")
    def test_infer_unet(self, mock_log_results, mock_unet_inference, mock_image_tensor, mock_artifact):
        """Test inference with UNet model."""
        # Setup mocks
        mock_unet_inference.return_value = np.random.rand(1, 1, 224, 224)

        # Create test weights file
        weights_path = Path("mock_artifact_dir") / "model.pt"
        weights_path.parent.mkdir(exist_ok=True)
        torch.save(torch.nn.Linear(10, 2).state_dict(), weights_path)

        # Setup artifact mock
        mock_artifact.download.return_value = "mock_artifact_dir"

        try:
            with patch("wandb.init"), patch("wandb.use_artifact", return_value=mock_artifact):
                engine = InferenceEngine("unet")
                result = engine.infer(mock_image_tensor, threshold=0.5)

            assert isinstance(result, np.ndarray)
            assert result.shape == (224, 224, 1)
            mock_unet_inference.assert_called_once()
            mock_log_results.assert_called_once()
        finally:
            # Cleanup
            if weights_path.exists():
                weights_path.unlink()
            if weights_path.parent.exists():
                weights_path.parent.rmdir()

    @patch("autovisionai.core.inference.fast_scnn_inference")
    @patch("autovisionai.core.inference.log_inference_results")
    def test_infer_fast_scnn(self, mock_log_results, mock_fast_scnn_inference, mock_image_tensor, mock_artifact):
        """Test inference with FastSCNN model."""
        # Setup mocks
        mock_fast_scnn_inference.return_value = np.random.rand(1, 1, 224, 224)

        # Create test weights file
        weights_path = Path("mock_artifact_dir") / "model.pt"
        weights_path.parent.mkdir(exist_ok=True)
        torch.save(torch.nn.Linear(10, 2).state_dict(), weights_path)

        # Setup artifact mock
        mock_artifact.download.return_value = "mock_artifact_dir"

        try:
            with patch("wandb.init"), patch("wandb.use_artifact", return_value=mock_artifact):
                engine = InferenceEngine("fast_scnn")
                result = engine.infer(mock_image_tensor, threshold=0.5)

            assert isinstance(result, np.ndarray)
            assert result.shape == (224, 224, 1)
            mock_fast_scnn_inference.assert_called_once()
            mock_log_results.assert_called_once()
        finally:
            # Cleanup
            if weights_path.exists():
                weights_path.unlink()
            if weights_path.parent.exists():
                weights_path.parent.rmdir()

    @patch("autovisionai.core.inference.mask_rcnn_inference")
    @patch("autovisionai.core.inference.log_inference_results")
    def test_infer_mask_rcnn(self, mock_log_results, mock_mask_rcnn_inference, mock_image_tensor, mock_artifact):
        """Test inference with Mask R-CNN model."""
        # Setup mocks
        mock_mask_rcnn_inference.return_value = (
            np.random.rand(1, 4),  # boxes
            np.array([1]),  # labels
            np.array([0.9]),  # scores
            [np.random.rand(1, 1, 224, 224)],  # masks
        )

        # Create test weights file
        weights_path = Path("mock_artifact_dir") / "model.pt"
        weights_path.parent.mkdir(exist_ok=True)
        torch.save(torch.nn.Linear(10, 2).state_dict(), weights_path)

        # Setup artifact mock
        mock_artifact.download.return_value = "mock_artifact_dir"

        try:
            with patch("wandb.init"), patch("wandb.use_artifact", return_value=mock_artifact):
                engine = InferenceEngine("mask_rcnn")
                result = engine.infer(mock_image_tensor, threshold=0.5)

            assert isinstance(result, np.ndarray)
            assert result.shape == (224, 224, 1)
            mock_mask_rcnn_inference.assert_called_once()
            mock_log_results.assert_called_once()
        finally:
            # Cleanup
            if weights_path.exists():
                weights_path.unlink()
            if weights_path.parent.exists():
                weights_path.parent.rmdir()

    @patch("wandb.init")
    @patch("wandb.use_artifact")
    def test_infer_unsupported_model(self, mock_use_artifact, mock_wandb_init, mock_image_tensor):
        """Test inference with unsupported model."""
        # Setup mocks
        mock_wandb_init.return_value = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.version = "v1"
        mock_use_artifact.return_value = mock_artifact

        # The test should fail before trying to load weights
        with pytest.raises(ValueError) as exc_info:
            engine = InferenceEngine("unsupported_model")
            engine.infer(mock_image_tensor)
        assert "Unsupported model 'unsupported_model'" in str(exc_info.value)
        mock_wandb_init.assert_called_once()
        # Weights should not be loaded for unsupported models
        mock_use_artifact.assert_not_called()


@pytest.mark.integration
class TestInferenceEngineIntegration:
    """Integration tests for the InferenceEngine class."""

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
                for item in self.test_dir.glob("**/*"):
                    if item.is_file():
                        item.unlink()
                self.test_dir.rmdir()
            except PermissionError:
                logger.warning(f"Could not delete directory: {self.test_dir}")

    @patch("wandb.init")
    @patch("wandb.use_artifact")
    @patch("autovisionai.core.inference.unet_inference")
    def test_full_inference_cycle(self, mock_unet_inference, mock_use_artifact, mock_wandb_init, mock_artifact):
        """Test a complete inference cycle with a mock model."""
        # Setup test environment
        CONFIG.logging.ml_loggers.root_dir = self.test_dir

        # Setup artifact mock
        mock_artifact.version = "v1"
        mock_artifact.download.return_value = "mock_artifact_dir"
        mock_use_artifact.return_value = mock_artifact

        # Setup inference mock
        mock_unet_inference.return_value = np.random.rand(1, 1, 224, 224)

        # Create minimal test weights file
        weights_path = Path("mock_artifact_dir") / "model.pt"
        weights_path.parent.mkdir(exist_ok=True)
        torch.save({"dummy": torch.tensor(1.0)}, weights_path)

        try:
            # Create and run inference engine
            engine = InferenceEngine("unet")
            image_tensor = torch.randn(1, 3, 224, 224)
            result = engine.infer(image_tensor, threshold=0.5)

            # Verify results
            assert isinstance(result, np.ndarray)
            assert result.shape == (224, 224, 1)

            # Verify inference was called
            mock_unet_inference.assert_called_once()
        finally:
            # Cleanup
            if weights_path.exists():
                weights_path.unlink()
            if weights_path.parent.exists():
                weights_path.parent.rmdir()


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down test environment."""
        # Reset ModelRegistry state before each test
        ModelRegistry._engines = {}
        ModelRegistry._initialized = False
        yield

    @patch("autovisionai.core.inference.InferenceEngine")
    def test_get_model(self, mock_inference_engine):
        """Test getting a model from the registry."""
        # Setup mock
        mock_engine = MagicMock()
        mock_inference_engine.return_value = mock_engine

        # Test getting a model
        engine = ModelRegistry.get_model("unet")
        assert engine == mock_engine
        mock_inference_engine.assert_called_once_with("unet")

        # Test getting the same model again (should return cached instance)
        engine2 = ModelRegistry.get_model("unet")
        assert engine2 == mock_engine
        assert mock_inference_engine.call_count == 1  # Should not create new instance

    def test_get_model_unsupported(self):
        """Test getting an unsupported model."""
        with pytest.raises(ValueError) as exc_info:
            ModelRegistry.get_model("unsupported_model")
        assert "not in" in str(exc_info.value)

    @patch("autovisionai.core.inference.InferenceEngine")
    @pytest.mark.asyncio
    async def test_initialize_models(self, mock_inference_engine):
        """Test initializing all models."""
        # Setup mock
        mock_engine = MagicMock()
        mock_inference_engine.return_value = mock_engine

        # Initialize models
        await ModelRegistry.initialize_models()

        # Verify all models were initialized
        assert ModelRegistry._initialized
        assert len(ModelRegistry._engines) == len(CONFIG.models.available)
        assert all(model in ModelRegistry._engines for model in CONFIG.models.available)
        assert mock_inference_engine.call_count == len(CONFIG.models.available)

    @patch("autovisionai.core.inference.InferenceEngine")
    @pytest.mark.asyncio
    async def test_initialize_models_idempotent(self, mock_inference_engine):
        """Test that initialize_models is idempotent."""
        # Setup mock
        mock_engine = MagicMock()
        mock_inference_engine.return_value = mock_engine

        # Initialize models twice
        await ModelRegistry.initialize_models()
        await ModelRegistry.initialize_models()

        # Verify models were only initialized once
        assert mock_inference_engine.call_count == len(CONFIG.models.available)
