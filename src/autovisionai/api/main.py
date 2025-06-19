import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from autovisionai.api.endpoints.inference import router as inference_router
from autovisionai.api.endpoints.train import router as train_router
from autovisionai.core.configs import ENV_MODE
from autovisionai.core.inference import ModelRegistry

logger = logging.getLogger(__name__)

logger.info(f"Starting AutoVisionAI API in {ENV_MODE} mode")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await ModelRegistry.initialize_models()
    yield
    # Shutdown
    pass


app = FastAPI(title="AutoVisionAI API", version="0.1.0", lifespan=lifespan)

app.include_router(inference_router)
app.include_router(train_router)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker containers"""
    return {"status": "healthy", "service": "autovisionai-api"}
