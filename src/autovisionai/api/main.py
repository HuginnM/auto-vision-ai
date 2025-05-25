from contextlib import asynccontextmanager

from fastapi import FastAPI

from autovisionai.api.endpoints.inference import router as inference_router
from autovisionai.core.inference import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await ModelRegistry.initialize_models()
    yield
    # Shutdown
    pass


app = FastAPI(title="AutoVisionAI API", version="0.1.0", lifespan=lifespan)

app.include_router(inference_router)
