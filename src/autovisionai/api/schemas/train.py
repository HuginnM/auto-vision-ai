from typing import List, Optional

from pydantic import BaseModel


class TrainingRequest(BaseModel):
    """Training request schema."""

    experiment_name: str
    model_name: str
    batch_size: Optional[int] = None
    epoch_patience: Optional[int] = None
    use_resize: Optional[bool] = None
    use_random_crop: Optional[bool] = None
    use_hflip: Optional[bool] = None
    max_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None
    weight_decay: Optional[float] = None
    scheduler_type: Optional[str] = None
    scheduler_step_size: Optional[int] = None
    scheduler_gamma: Optional[float] = None


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
