from typing import Optional

from autovisionai.core.inference import InferenceEngine
from autovisionai.core.utils.utils import get_input_image_for_inference


def run_inference_service(model_name: str, image_path: Optional[str] = None, image_url: Optional[str] = None) -> dict:
    try:
        if image_path:
            image_tensor = get_input_image_for_inference(local_path=image_path)
        elif image_url:
            image_tensor = get_input_image_for_inference(url=image_url)
        else:
            return {
                "status": "error",
                "detail": "No image_path or image_url provided.",
                "mask_shape": None,
            }
        engine = InferenceEngine(model_name)
        mask = engine.infer(image_tensor, return_binary=True)
        return {
            "status": "success",
            "detail": "Inference completed.",
            "mask_shape": list(mask.shape),
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
            "mask_shape": None,
        }
