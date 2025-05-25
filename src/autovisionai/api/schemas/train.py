from typing import Optional

from pydantic import BaseModel


class TrainingRequest(BaseModel):
    experiment_name: str
    model_name: str
    batch_size: int = 4
    epoch_patience: int = 2
    use_resize: bool = False
    use_random_crop: bool = False
    use_hflip: bool = False
    max_epochs: Optional[int] = None


class TrainingResponse(BaseModel):
    status: str
    detail: str
    experiment_path: Optional[str] = None
    model_weights_path: Optional[str] = None
