import base64
import io
import zlib

import numpy as np
from PIL import Image


def encode_array_to_base64(arr: np.ndarray) -> str:
    """Encode a numpy array to a base64 string.

    Args:
        arr: Numpy array to encode.

    Returns:
        Base64 encoded string.
    """
    buf = io.BytesIO()
    np.save(buf, arr)
    compressed = zlib.compress(buf.getvalue())
    return base64.b64encode(compressed).decode("utf-8")


def decode_array_from_base64(b64_str: str) -> np.ndarray:
    """Decode a base64 string to a numpy array.

    Args:
        b64_str: Base64 encoded string.

    Returns:
        Numpy array.
    """
    compressed = base64.b64decode(b64_str)
    arr_bytes = zlib.decompress(compressed)
    return np.load(io.BytesIO(arr_bytes))


def encode_image_to_base64(img: Image.Image, format="PNG") -> str:
    """Encode a PIL image to a base64 string.

    Args:
        img: PIL image to encode.
        format: Image format to save as. Defaults to 'PNG'.

    Returns:
        Base64 encoded string.
    """
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_image_from_base64(b64_str: str) -> Image.Image:
    """Decode a base64 string to a PIL image.

    Args:
        b64_str: Base64 encoded string.

    Returns:
        PIL image.
    """
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))
