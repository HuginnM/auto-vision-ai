import logging
from pathlib import Path

import numpy as np
import torch

import wandb
from autovisionai.core.configs import CONFIG
from autovisionai.core.loggers.ml_logging import log_inference_results
from autovisionai.core.models.fast_scnn.fast_scnn_inference import model_inference as fast_scnn_inference
from autovisionai.core.models.mask_rcnn.mask_rcnn_inference import model_inference as mask_rcnn_inference
from autovisionai.core.models.unet.unet_inference import model_inference as unet_inference
from autovisionai.core.utils.utils import show_pic_and_pred_semantic_mask

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.run = wandb.init(project="autovisionai_inference", entity="arthur-sobol-private")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.weights = self._load_weights_from_wandb()

    def _load_weights_from_wandb(self) -> dict:
        """
        Load best weights for the given model from W&B Artifacts directly into memory.
        """
        logger.info(f"Fetching best weights for '{self.model_name}' from W&B")

        self.artifact_ref = f"wandb-registry-model/AutoVisionAI:{self.model_name}"

        try:
            artifact = wandb.use_artifact(self.artifact_ref, type="model", aliases=["production", self.model_name])
        except Exception as e:
            logger.error(f"Failed to load artifact {self.artifact_ref}: {str(e)}")

        self.model_version = artifact.version

        # Download artifact directly to memory
        artifact_dir = artifact.download()

        weights_path = Path(artifact_dir) / "model.pt"

        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found in artifact at {weights_path}")

        logger.info(f"Loaded weights from {self.artifact_ref} to {self.device}")
        return weights_path

    def infer(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Run inference on the input image and log results to all enabled ML loggers.
        """
        logger.info(f"Running inference for '{self.model_name}' on {self.device}")

        import time

        start_time = time.time()

        # Run inference with appropriate model
        match self.model_name:
            case "unet":
                image_mask = unet_inference(image=image_tensor, trained_model_path=self.weights)
            case "fast_scnn":
                image_mask = fast_scnn_inference(image=image_tensor, trained_model_path=self.weights)
            case "mask_rcnn":
                image_mask = mask_rcnn_inference(image=image_tensor, trained_model_path=self.weights)
            case _:
                raise ValueError(
                    f"Unsupported model '{self.model_name}'.\nAvailable models: {CONFIG.models.available}."
                )

        inference_time = time.time() - start_time
        return image_mask
        # Log results using native APIs
        log_inference_results(
            input_image=image,
            output_mask=image_mask,
            model_name=self.model_name,
            model_version=self.model_version,
            inference_time=inference_time,
        )

        return image_mask

    def __del__(self):
        """
        Cleanup: finish the W&B run when the engine is destroyed
        """
        if hasattr(self, "run"):
            self.run.finish()


def main(model: str, image=None):
    from autovisionai.core.utils.utils import show_pic_and_pred_semantic_mask

    infer_engine = InferenceEngine(model)
    image_mask = infer_engine.infer(image)
    show_pic_and_pred_semantic_mask(image, image_mask, threshold=0.5, use_plt=False)


if __name__ == "__main__":
    # Example usage
    from autovisionai.core.utils.utils import get_input_image_for_inference

    model = "fast_scnn"
    image = get_input_image_for_inference(local_path=r"C:\DATA\Projects\AutoVisionAI\data\images\0cdf5b5d0ce1_04.jpg")
    engine = InferenceEngine(model)
    pred_mask = engine.infer(image)
    show_pic_and_pred_semantic_mask(image, pred_mask, threshold=0.5, use_plt=False)
