from fastapi import FastAPI
from src.autovisionai.api.endpoints.inference import router as inference_router

app = FastAPI(title="AutoVisionAI API", version="0.1.0")

app.include_router(inference_router)
