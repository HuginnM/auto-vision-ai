import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from src.autovisionai.api.schemas.inference import InferenceResponse
from src.autovisionai.api.services.inference_service import run_inference_service

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/", response_model=InferenceResponse)
async def inference_endpoint(
    model_name: str = Form(...),  # noqa: B008
    file: Optional[UploadFile] = File(None),  # noqa: B008
    image_url: Optional[str] = Form(None),  # noqa: B008
):
    tmp_path = None
    result = None
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        try:
            result = run_inference_service(model_name, image_path=tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    elif image_url is not None:
        result = run_inference_service(model_name, image_url=image_url)
    else:
        raise HTTPException(status_code=400, detail="Either file or image_url must be provided.")
    return InferenceResponse(**result)
