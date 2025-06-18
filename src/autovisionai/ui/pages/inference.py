"""Inference page for AutoVisionAI Streamlit UI."""

import io

import numpy as np
import requests
import streamlit as st
from PIL import Image

from autovisionai.core.configs import CONFIG
from autovisionai.core.utils.encoding import decode_array_from_base64
from autovisionai.core.utils.utils import apply_mask_to_image
from autovisionai.ui.utils import configure_sidebar, format_model_name


def get_model_info(model_name: str) -> dict:
    """Get information about a model."""
    model_info = {
        "unet": {
            "name": "U-Net",
            "description": "Classic encoder-decoder architecture for semantic segmentation",
            "params": "~31M",
            "speed": "Medium",
        },
        "fast_scnn": {
            "name": "Fast-SCNN",
            "description": "Fast Segmentation CNN optimized for real-time inference",
            "params": "~1.1M",
            "speed": "Fast",
        },
        "mask_rcnn": {
            "name": "Mask R-CNN",
            "description": "Instance segmentation with object detection capabilities",
            "params": "~44M",
            "speed": "Slow",
        },
    }
    return model_info.get(
        model_name, {"name": model_name, "description": "Unknown model", "params": "Unknown", "speed": "Unknown"}
    )


def create_placeholder_mask(width: int, height: int) -> Image.Image:
    """Create a placeholder segmentation mask for demonstration."""
    # Create a simple car-shaped mask in the center
    mask_array = np.zeros((height, width), dtype=np.uint8)

    # Create a simple rectangular mask in the center (representing a car)
    center_x, center_y = width // 2, height // 2
    car_width, car_height = min(width // 3, 150), min(height // 3, 100)

    x1 = center_x - car_width // 2
    x2 = center_x + car_width // 2
    y1 = center_y - car_height // 2
    y2 = center_y + car_height // 2

    # Ensure coordinates are within bounds
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)

    mask_array[y1:y2, x1:x2] = 255

    # Convert to RGB for display
    mask_rgb = np.stack([mask_array, mask_array, mask_array], axis=2)

    return Image.fromarray(mask_rgb)


def run_inference(image: Image.Image, image_source: str, source_data, model_name: str):
    """Run inference using the API."""
    with st.spinner("Running inference..."):
        try:
            api_url = st.session_state.get("api_base_url", CONFIG.app.api_base_url)
            inference_url = f"{api_url}/inference/"

            if image_source == "file":
                # Prepare file for upload
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                img_bytes.seek(0)

                files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                data = {"model_name": model_name}

                # Extended timeout for the first request to let the model load
                response = requests.post(inference_url, files=files, data=data, timeout=60)

            else:  # URL
                data = {"model_name": model_name, "image_url": source_data}
                response = requests.post(inference_url, data=data, timeout=30)

            response.raise_for_status()
            result = response.json()

            # Store result in session state
            st.session_state.inference_result = {"result": result, "image": image, "model_name": model_name}

            st.success("Inference completed successfully!")
            return True

        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to API at {api_url}. Make sure the API server is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        return False


def display_results(result_data: dict):
    """Display inference results."""
    result = result_data["result"]
    image = result_data["image"]
    model_name = result_data["model_name"]

    try:
        mask_array = decode_array_from_base64(result["mask_data"])
        st.success(f"Status: {result['status']}")
    except Exception as e:
        mask_array = None
        st.warning(f"Could not convert mask data: {e}")
        st.error(f"Error details: {str(e)}")

    # Display result details
    with st.expander("📋 Result Details", expanded=True):
        st.write(f"**Model:** {model_name}")
        st.write(f"**Status:** {result['status']}")
        st.write(f"**Detail:** {result['detail']}")

        # Display mask info if available
        if mask_array is not None:
            st.write(f"**Mask Shape:** {mask_array.shape}")

            # Display mask shape as a metric
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Height", mask_array.shape[0])
            with col2:
                st.metric("Width", mask_array.shape[1])

    # Mask visualization with actual mask data
    if mask_array is not None:
        st.subheader("🎭 Segmentation Mask")

        # Display original and mask side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("**Segmentation Mask**")
            binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
            mask_image = Image.fromarray(binary_mask, mode="L")
            st.image(mask_image, use_container_width=True)

        # Overlay visualization
        st.markdown("**Overlay Visualization**")
        overlay_image = apply_mask_to_image(image, mask_array, 0.5)
        st.image(overlay_image, use_container_width=True)

    # Download results section
    st.subheader("💾 Download Results")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Download Report", use_container_width=True, key="download_report_inference"):
            # Create report
            results_text = f"""AutoVisionAI Inference Results
================================
Model: {model_name}
Status: {result["status"]}
Detail: {result["detail"]}
Mask Shape: {result.get("mask_shape", "N/A")}
Has Mask Data: {bool(result.get("mask_data"))}
"""
            st.download_button(
                label="📄 Download Results Report",
                data=results_text,
                file_name=f"inference_results_{model_name}.txt",
                mime="text/plain",
                key="download_report_file_inference",
            )

    with col2:
        if st.button("🖼️ Download Images", use_container_width=True, key="download_images_inference"):
            # Download mask as PNG if available
            if result.get("mask_data") and mask_image:
                try:
                    # Convert mask image to bytes
                    img_bytes = io.BytesIO()
                    mask_image.save(img_bytes, format="PNG")
                    img_bytes.seek(0)

                    st.download_button(
                        label="💾 Download Mask PNG",
                        data=img_bytes.getvalue(),
                        file_name=f"mask_{model_name}.png",
                        mime="image/png",
                        key="download_mask_file",
                    )
                except Exception as e:
                    st.error(f"Could not prepare mask download: {e}")
            else:
                st.info("Mask download available when inference returns actual mask data")


# Main page content
st.title("🔍 Inference")
st.markdown("Upload an image or provide a URL to run car segmentation inference.")

configure_sidebar()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input")

    # Input method selection
    input_method = st.radio(
        "Choose input method:", ["Upload File", "Image URL"], horizontal=True, key="input_method_radio"
    )

    image = None
    image_source = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload an image file (PNG, JPG, JPEG)",
            key="image_file_uploader",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_source = "file"
            st.image(image, caption="Uploaded Image", use_container_width=True)

    else:  # Image URL
        image_url = st.text_input(
            "Image URL",
            placeholder="https://example.com/image.jpg",
            help="Enter the URL of an image",
            key="image_url_input",
        )

        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                image_source = "url"
                st.image(image, caption="Image from URL", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load image from URL: {e}")

    model_name = st.selectbox(
        "🤖 Select Model",
        ["unet", "fast_scnn", "mask_rcnn"],
        index=0,
        help="Choose the model architecture for inference",
        key="inference_model_select",
        format_func=format_model_name,
    )

    # Model info
    model_info = get_model_info(model_name)
    with st.expander("📋 Model Information", expanded=False):
        st.write(f"**Name:** {model_info['name']}")
        st.write(f"**Description:** {model_info['description']}")
        st.write(f"**Parameters:** {model_info['params']}")
        st.write(f"**Speed:** {model_info['speed']}")

    # Run inference button
    if image is not None and st.button(
        "🚀 Run Inference", type="primary", use_container_width=True, key="run_inference_btn"
    ):
        source_data = uploaded_file if image_source == "file" else image_url
        if run_inference(image, image_source, source_data, model_name):
            # Results will be displayed in the right column
            pass

with col2:
    st.subheader("📊 Results")
    if "inference_result" not in st.session_state:
        st.info("Upload an image and click 'Run Inference' to see results.")
    else:
        display_results(st.session_state.inference_result)
