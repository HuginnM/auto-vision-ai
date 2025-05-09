import logging
from pathlib import Path

import numpy as np
import wandb
from PIL import Image

from autovisionai.core.models.fast_scnn.fast_scnn_inference import model_inference as fast_scnn_inference
from autovisionai.core.models.mask_rcnn.mask_rcnn_inference import model_inference as mask_rcnn_inference
from autovisionai.core.models.unet.unet_inference import model_inference as unet_inference

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.weights_path = self._load_weights_from_wandb()

    def _load_weights_from_wandb(self) -> Path:
        """
        Load best weights for the given model from W&B Artifacts and cache locally.
        """
        logger.info(f"Fetching best weights for '{self.model_name}' from W&B")

        artifact_ref = f"{self.model_name}:production"
        artifact = wandb.use_artifact(artifact_ref, type="model")
        artifact_dir = artifact.download()

        weights_path = Path(artifact_dir) / "production.pth"

        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found at {weights_path}")

        logger.info(f"Loaded weights from {weights_path}")
        return weights_path

    def infer(self, image: Image.Image) -> np.ndarray:
        logger.info(f"Running inference for '{self.model_name}'")

        match self.model_name:
            case "unet":
                return unet_inference(image=image, weights_path=self.weights_path)
            case "fast_scnn":
                return fast_scnn_inference(image=image, weights_path=self.weights_path)
            case "mask_rcnn":
                return mask_rcnn_inference(image=image, weights_path=self.weights_path)
            case _:
                raise ValueError(f"Unsupported model '{self.model_name}'")
