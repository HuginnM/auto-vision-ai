from typing import List, Optional

from pydantic import BaseModel


class InferenceResponse(BaseModel):
    status: str
    detail: str
    mask_shape: Optional[List[int]] = None
