import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from autovisionai.api.schemas.train import TrainingProgress, TrainingRequest, TrainingResponse
from autovisionai.api.services.train_service import training_service
from autovisionai.api.services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])
manager = WebSocketManager()  # Create WebSocket manager instance


@router.post("/", response_model=TrainingResponse)
async def train_endpoint(request: TrainingRequest):
    """Start a new training job.

    Args:
        request (TrainingRequest): Training configuration

    Returns:
        TrainingResponse: Initial response with training status
    """
    try:
        # Create WebSocket callback function
        async def progress_callback(progress: TrainingProgress):
            # Send progress update through WebSocket
            await manager.broadcast(progress.__dict__)

        # Start training in background with progress callback
        result = await training_service.train_model(request, progress_callback)
        return TrainingResponse(**result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": str(e),
                "experiment_path": None,
                "model_weights_path": None,
            },
        )


@router.websocket("/ws/{experiment_name}")
async def training_progress_websocket(websocket: WebSocket, experiment_name: str):
    """WebSocket endpoint for training progress updates.

    Args:
        websocket (WebSocket): WebSocket connection
        experiment_name (str): Name of the experiment to track
    """
    logger.info(f"WebSocket connection request for experiment: {experiment_name}")

    try:
        # Accept the connection first
        await manager.connect(websocket)
        logger.info(f"WebSocket connection accepted for experiment: {experiment_name}")

        # Wait for training to start (up to 30 seconds)
        for _ in range(30):
            if experiment_name in training_service.active_trainings:
                logger.info(f"Training {experiment_name} found in active trainings")
                break
            await asyncio.sleep(1)
        else:
            logger.warning(f"Training {experiment_name} did not start within timeout")
            return

        # Keep connection alive while training is active
        while True:
            if experiment_name not in training_service.active_trainings:
                # Give a small delay to ensure final update is sent
                await asyncio.sleep(0.5)
                break
            await asyncio.sleep(1)

        logger.info(f"Training {experiment_name} completed or not found in active trainings")
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for experiment: {experiment_name}")
    finally:
        await manager.disconnect(websocket)  # Ensure we always disconnect
