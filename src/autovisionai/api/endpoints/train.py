import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from autovisionai.api.schemas.train import TrainingRequest, TrainingResponse
from autovisionai.api.services.train_service import training_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/", response_model=TrainingResponse)
async def train_endpoint(request: TrainingRequest):
    """Start a new training job.

    Args:
        request (TrainingRequest): Training configuration

    Returns:
        TrainingResponse: Initial response with training status
    """
    try:
        # Start training in background
        result = await training_service.train_model(request)
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
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for experiment: {experiment_name}")

    try:
        while True:
            progress = training_service.get_training_progress(experiment_name)
            if progress:
                logger.info(f"Sending progress update: {progress.__dict__}")
                await websocket.send_json(
                    {
                        "current_epoch": progress.current_epoch,
                        "total_epochs": progress.total_epochs,
                        "current_loss": progress.current_loss,
                        "best_loss": progress.best_loss,
                        "status": progress.status,
                        "detail": progress.detail,
                    }
                )
                if progress.status in ["completed", "error"]:
                    logger.info(f"Training {progress.status}, closing WebSocket")
                    break
            else:
                logger.debug(f"No progress found for experiment: {experiment_name}")
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        await websocket.send_json(
            {
                "status": "error",
                "detail": str(e),
            }
        )
