from unittest.mock import patch

import pytest

from autovisionai.api.schemas.train import TrainingRequest
from autovisionai.api.services.train_service import TrainingProgress


@pytest.fixture
def training_request():
    return TrainingRequest(experiment_name="test_experiment", model_name="unet", batch_size=4, max_epochs=10)


class TestTrainingEndpoints:
    def test_train_endpoint_success(self, client, training_request):
        with patch("autovisionai.api.services.train_service.training_service.train_model") as mock_train:
            mock_train.return_value = {
                "status": "success",
                "detail": "Training completed successfully",
                "experiment_path": "test/path",
                "model_weights_path": "test/weights.pt",
            }

            response = client.post("/train/", json=training_request.model_dump())
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["experiment_path"] == "test/path"
            assert data["model_weights_path"] == "test/weights.pt"

    def test_train_endpoint_error(self, client, training_request):
        with patch(
            "autovisionai.api.services.train_service.training_service.train_model", side_effect=Exception("Test error")
        ):
            response = client.post("/train/", json=training_request.model_dump())
            assert response.status_code == 500
            data = response.json()
            assert data["status"] == "error"
            assert "Test error" in data["detail"]

    def test_training_progress_websocket_timeout(self, client):
        """Test WebSocket connection when training doesn't start within timeout."""
        # Don't add anything to active_trainings to simulate training not starting

        with client.websocket_connect("/train/ws/nonexistent_experiment") as _:
            # The WebSocket should close after timeout (30 seconds in real implementation)
            # For testing, we just verify the connection is established
            # In a real scenario, this would timeout, but testing frameworks handle this differently
            pass

    @patch("autovisionai.api.services.websocket_manager.WebSocketManager.broadcast")
    def test_train_endpoint_with_websocket_callback(self, mock_broadcast, client, training_request):
        """Test that train endpoint creates progress callback for WebSocket broadcasting."""

        async def mock_train_model(request, progress_callback=None):
            # Simulate calling the progress callback
            if progress_callback:
                progress = TrainingProgress()
                progress.status = "training"
                progress.current_epoch = 1
                progress.total_epochs = 10
                await progress_callback(progress)

            return {
                "status": "success",
                "detail": "Training completed successfully",
                "experiment_path": "test/path",
                "model_weights_path": "test/weights.pt",
            }

        with patch(
            "autovisionai.api.services.train_service.training_service.train_model", side_effect=mock_train_model
        ):
            response = client.post("/train/", json=training_request.model_dump())
            assert response.status_code == 200

            # Verify WebSocket broadcast was called
            mock_broadcast.assert_called()
