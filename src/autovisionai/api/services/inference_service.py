from typing import Optional

from autovisionai.core.inference import ModelRegistry
from autovisionai.core.utils.encoding import encode_array_to_base64
from autovisionai.core.utils.utils import get_input_image_for_inference


def run_inference_service(model_name: str, image_path: Optional[str] = None, image_url: Optional[str] = None) -> dict:
    """Run inference using the specified model on an input image.

    Args:
        model_name (str): Name of the model to use for inference.
        image_path (Optional[str], optional): Local path to input image file. Defaults to None.
        image_url (Optional[str], optional): URL of input image. Defaults to None.

    Returns:
        dict: Dictionary containing:
            - status (str): "success" or "error"
            - detail (str): Description of the result or error message
            - mask_data (Optional[List[List[int]]]): Binary mask as 2D list
    """
    try:
        if image_path:
            image_tensor = get_input_image_for_inference(local_path=image_path)
        elif image_url:
            image_tensor = get_input_image_for_inference(url=image_url)
        else:
            return {
                "status": "error",
                "detail": "No image_path or image_url provided.",
                "mask_data": None,
            }

        engine = ModelRegistry.get_model(model_name)
        mask = engine.infer(image_tensor)

        return {
            "status": "success",
            "detail": "Inference completed.",
            "mask_data": encode_array_to_base64(mask),
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
            "mask_data": None,
        }
