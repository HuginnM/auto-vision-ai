from typing import List, Optional

from pydantic import BaseModel


class TrainingRequest(BaseModel):
    """Training request schema."""

    experiment_name: str
    model_name: str
    batch_size: int = 4
    epoch_patience: int = 2
    use_resize: bool = False
    use_random_crop: bool = False
    use_hflip: bool = False
    max_epochs: Optional[int] = None


class TrainingResponse(BaseModel):
    """Training response schema."""

    status: str
    detail: str
    experiment_path: Optional[str] = None
    model_weights_path: Optional[str] = None


class TrainingProgress(BaseModel):
    """Training progress schema."""

    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = float("inf")
    best_loss: float = float("inf")
    status: str = "initializing"
    detail: str = ""
    output_logs: List[str] = []
