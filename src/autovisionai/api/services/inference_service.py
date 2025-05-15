from autovisionai.core.inference import InferenceEngine
from autovisionai.core.utils.utils import get_input_image_for_inference


def run_inference_service(model_name: str, image_path: str) -> dict:
    try:
        image_tensor = get_input_image_for_inference(local_path=image_path)
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
