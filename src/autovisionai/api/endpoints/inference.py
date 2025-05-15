import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, UploadFile
from src.autovisionai.api.schemas.inference import InferenceResponse
from src.autovisionai.api.services.inference_service import run_inference_service

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/", response_model=InferenceResponse)
async def inference_endpoint(model_name: str = Form(...), file: UploadFile = File(...)):  # noqa: B008
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_inference_service(model_name, tmp_path)
        return InferenceResponse(**result)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
