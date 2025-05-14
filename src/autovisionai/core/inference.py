import logging
import os
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
        self.run = wandb.init(project="autovisionai_inference", entity=os.getenv("WANDB_ENTITY"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.weights = None
        self.model_version = None

    def _load_weights_from_wandb(self) -> dict:
        """
        Load best weights for the given model from W&B Artifacts directly into memory.
        """
        logger.info(f"Fetching best weights for '{self.model_name}' from W&B")

        self.artifact_ref = f"wandb-registry-model/{self.model_name}:production"

        try:
            artifact = wandb.use_artifact(self.artifact_ref, type="model")
            self.model_version = artifact.version
            artifact_dir = artifact.download()
        except Exception as e:
            logger.error(f"Failed to load artifact {self.artifact_ref}: {str(e)}")
            raise  # Re-raise the exception to be handled by the caller

        weights_path = Path(artifact_dir) / "model.pt"

        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found in artifact at {weights_path}")

        logger.info(f"Loaded weights from {self.artifact_ref} to {self.device}")
        return weights_path

    def process_results(self, image_tensor: torch.Tensor, raw_mask: np.ndarray, threshold: float = 0.5) -> tuple:
        """
        Process the raw inference results for logging and visualization.

        Args:
            image_tensor: Input image tensor (B, C, H, W)
            raw_mask: Raw prediction mask (B, C, H, W) or (B, H, W)
            threshold: Threshold for binary mask conversion

        Returns:
            tuple: (processed_image, processed_mask, binary_mask)
        """
        # Convert image tensor to numpy and handle dimensions
        processed_image = image_tensor.cpu().numpy().squeeze(0)  # Remove batch dim
        processed_image = np.transpose(processed_image, (1, 2, 0))  # Convert to (H, W, C)

        # Ensure mask is in the correct format
        if isinstance(raw_mask, torch.Tensor):
            raw_mask = raw_mask.cpu().numpy()
        raw_mask = raw_mask.squeeze(0)  # Remove batch dim

        # Handle different mask dimensions
        if raw_mask.ndim == 3:  # If mask has channels (C, H, W)
            raw_mask = np.transpose(raw_mask, (1, 2, 0))  # Convert to (H, W, C)
        elif raw_mask.ndim == 2:  # If mask is (H, W)
            raw_mask = np.expand_dims(raw_mask, axis=-1)  # Add channel dim (H, W, 1)

        # Create binary mask for visualization
        binary_mask = (raw_mask > threshold).astype(np.uint8) * 255

        return processed_image, raw_mask, binary_mask

    def infer(self, image_tensor: torch.Tensor, threshold: float = 0.5, return_binary: bool = False) -> np.ndarray:
        """
        Run inference on the input image and log results to all enabled ML loggers.

        Args:
            image_tensor: Input image tensor
            threshold: Threshold for binary mask conversion
            return_binary: If True, returns the processed binary mask instead of raw mask

        Returns:
            np.ndarray: Either raw mask or processed binary mask depending on return_processed flag
        """
        logger.info(f"Running inference for '{self.model_name}' on {self.device}")

        # Check if model is supported before loading weights
        if self.model_name not in CONFIG.models.available:
            raise ValueError(f"Unsupported model '{self.model_name}'.\nAvailable models: {CONFIG.models.available}.")

        # Load weights if not already loaded
        if self.weights is None:
            self.weights = self._load_weights_from_wandb()

        import time

        start_time = time.time()

        # Run inference with appropriate model
        match self.model_name:
            case "unet":
                image_mask = unet_inference(image=image_tensor, trained_model_path=self.weights)
            case "fast_scnn":
                image_mask = fast_scnn_inference(image=image_tensor, trained_model_path=self.weights)
            case "mask_rcnn":
                _, _, _, image_masks = mask_rcnn_inference(image=image_tensor, trained_model_path=self.weights)
                image_mask = image_masks[0]
            case _:
                raise ValueError(
                    f"Unsupported model '{self.model_name}'.\nAvailable models: {CONFIG.models.available}."
                )
        inference_time = time.time() - start_time  # Stop timer

        # Process results for logging
        processed_image, processed_mask, binary_mask = self.process_results(image_tensor, image_mask, threshold)

        # Log results using ML loggers
        log_inference_results(
            input_image=processed_image,
            output_mask=binary_mask,
            model_name=self.model_name,
            model_version=self.model_version,
            inference_time=inference_time,
        )

        return binary_mask if return_binary else processed_mask


def main(model: str, image=None):
    from autovisionai.core.utils.utils import show_pic_and_pred_semantic_mask

    infer_engine = InferenceEngine(model)
    image_mask = infer_engine.infer(image, return_binary=True)
    show_pic_and_pred_semantic_mask(image, image_mask, threshold=0.5, use_plt=False)


if __name__ == "__main__":
    # Example usage
    from autovisionai.core.utils.utils import get_input_image_for_inference

    model = "mask_rcnn"
    image = get_input_image_for_inference(local_path=r"C:\DATA\Projects\AutoVisionAI\data\images\0cdf5b5d0ce1_04.jpg")
    engine = InferenceEngine(model)
    pred_mask = engine.infer(image, return_binary=False)
    show_pic_and_pred_semantic_mask(image, pred_mask, threshold=0.5, use_plt=False)
