from unittest.mock import patch

import pytest

from autovisionai.api.schemas.train import TrainingRequest
from autovisionai.api.services.train_service import TrainingProgress, training_service


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

    def test_training_progress_websocket(self, client):
        progress = TrainingProgress()
        progress.current_epoch = 5
        progress.total_epochs = 10
        progress.current_loss = 0.5
        progress.best_loss = 0.4
        progress.status = "training"
        progress.detail = "Training in progress"

        training_service.active_trainings["test_experiment"] = progress

        with client.websocket_connect("/train/ws/test_experiment") as websocket:
            data = websocket.receive_json()
            assert data["current_epoch"] == 5
            assert data["total_epochs"] == 10
            assert data["current_loss"] == 0.5
            assert data["best_loss"] == 0.4
            assert data["status"] == "training"
            assert data["detail"] == "Training in progress"

    def test_training_progress_websocket_completed(self, client):
        progress = TrainingProgress()
        progress.status = "completed"
        progress.detail = "Training completed"

        training_service.active_trainings["test_experiment"] = progress

        with client.websocket_connect("/train/ws/test_experiment") as websocket:
            data = websocket.receive_json()
            assert data["status"] == "completed"
            assert data["detail"] == "Training completed"
