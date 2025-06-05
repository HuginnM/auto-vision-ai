from typing import Optional

from pydantic import BaseModel


class InferenceResponse(BaseModel):
    status: str
    detail: str
    mask_data: Optional[str] = None  # Base64 encoded mask data
