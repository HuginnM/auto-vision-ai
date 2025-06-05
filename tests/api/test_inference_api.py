#!/usr/bin/env python3
"""Test script for inference API endpoint."""

from io import BytesIO

import numpy as np
import requests
from PIL import Image


def test_inference_endpoint():
    """Test the inference endpoint with a sample image."""
    # Create a simple test image
    test_img = Image.new("RGB", (256, 256), color="blue")

    # Save to bytes
    img_bytes = BytesIO()
    test_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Test the API
    api_url = "http://localhost:8000"
    endpoint = f"{api_url}/inference/"

    try:
        # Test with file upload
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        data = {"model_name": "unet"}

        print("🔍 Testing inference endpoint...")
        response = requests.post(endpoint, files=files, data=data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("API Response received!")
            print(f"Status: {result.get('status')}")
            print(f"Detail: {result.get('detail')}")
            print(f"Mask Shape: {result.get('mask_shape')}")
            print(f"Has mask_data: {bool(result.get('mask_data'))}")

            # Test mask data
            if result.get("mask_data"):
                mask_array = np.array(result["mask_data"])
                print(f"Mask data shape: {mask_array.shape}")
                print(f"Mask data type: {mask_array.dtype}")
                print(f"Unique values: {np.unique(mask_array)}")
                print(f"Binary mask (0/1): {set(np.unique(mask_array)).issubset({0, 1})}")
            else:
                print("⚠️ No mask_data returned")

            return True
        else:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("Could not connect to API. Make sure the server is running at http://localhost:8000")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing AutoVisionAI Inference API")
    print("=" * 50)

    success = test_inference_endpoint()

    if success:
        print("\nAll tests passed!")
    else:
        print("\nTests failed!")
        print("Make sure:")
        print("1. API server is running: uvicorn src.autovisionai.api.main:app --reload")
        print("2. Models are properly loaded")
        print("3. Dependencies are installed")
