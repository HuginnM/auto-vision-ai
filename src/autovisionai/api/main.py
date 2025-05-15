from fastapi import FastAPI

from autovisionai.api.endpoints.inference import router as inference_router
from autovisionai.core.inference import ModelRegistry

app = FastAPI(title="AutoVisionAI API", version="0.1.0")

app.include_router(inference_router)


@app.on_event("startup")
async def warm_up_models() -> None:
    # Pre-load all models concurrently
    await ModelRegistry.initialize_models()
