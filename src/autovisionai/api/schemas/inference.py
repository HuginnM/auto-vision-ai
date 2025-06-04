from pydantic import BaseModel


class InferenceResponse(BaseModel):
    status: str
    detail: str
    mask_data: str = None
