import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from autovisionai.api.schemas.inference import InferenceResponse
from autovisionai.api.services.inference_service import run_inference_service

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/", response_model=InferenceResponse)
async def inference_endpoint(
    model_name: str = Form(...),  # noqa: B008
    file: Optional[UploadFile] = File(None),  # noqa: B008
    image_url: Optional[str] = Form(None),  # noqa: B008
):
    """Run inference on an image using a specified model.

    Args:
        model_name (str): Name of the model to use for inference
        file (Optional[UploadFile], optional): Image file upload. Defaults to None.
        image_url (Optional[str], optional): URL of image to process. Defaults to None.

    Returns:
        InferenceResponse: Response containing inference status, details and mask shape

    Raises:
        HTTPException: If neither file nor image_url is provided
    """
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
