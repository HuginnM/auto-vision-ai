"""Utility functions for AutoVisionAI UI."""

import threading
from typing import Callable, Optional

import requests
import streamlit as st
import websocket

from autovisionai.core.configs import CONFIG, PROJECT_ROOT, WANDB_ENTITY
from autovisionai.core.configs.schema import MLLoggersConfig
from autovisionai.core.utils.encoding import encode_image_path_to_base64


class WebSocketClient:
    """WebSocket client for training progress updates."""

    def __init__(self):
        self.ws = None
        self.is_connected = False
        self.message_callback: Optional[Callable] = None

    def connect(self, url: str, on_message: Callable, on_error: Callable = None, on_close: Callable = None):
        """Connect to WebSocket server."""
        try:

            def _on_message(ws, message):
                if self.message_callback:
                    self.message_callback(message)

            def _on_error(ws, error):
                self.is_connected = False
                if on_error:
                    on_error(error)

            def _on_close(ws, close_status_code, close_msg):
                self.is_connected = False
                if on_close:
                    on_close(close_status_code, close_msg)

            def _on_open(ws):
                self.is_connected = True

            self.message_callback = on_message
            self.ws = websocket.WebSocketApp(
                url, on_open=_on_open, on_message=_on_message, on_error=_on_error, on_close=_on_close
            )

            # Run in separate thread
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()

        except Exception as e:
            if on_error:
                on_error(str(e))

    def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.ws:
            self.ws.close()
            self.is_connected = False


def format_loss(loss: float) -> str:
    """Format loss value for display."""
    if loss == float("inf"):
        return "N/A"
    elif loss < 0.001:
        return f"{loss:.2e}"
    else:
        return f"{loss:.6f}"


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def validate_api_url(url: str) -> bool:
    """Validate API URL format."""
    import re

    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(pattern, url))


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


def check_api_endpoint(endpoint: str) -> bool:
    """Check if an API endpoint is accessible."""
    try:
        api_url = st.session_state.get("api_base_url", CONFIG.app.api_base_url)
        response = requests.options(f"{api_url}{endpoint}", timeout=2)
        return response.status_code in (200, 204, 405)
    except requests.RequestException:
        return False


def add_api_status():
    """Add API status sidebar with manual check to avoid page load delays."""
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = CONFIG.app.api_base_url

    st.subheader("⚙️ API Settings")
    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="Base URL of the AutoVisionAI API",
        key="global_api_url",
    )
    st.session_state.api_base_url = api_url

    st.markdown("**Connection Status:**")

    # Manual check button
    if st.button("🔄 Check API Connection"):
        with st.spinner("Checking..."):
            try:
                response = requests.get(f"{api_url}/docs", timeout=2)
                if response.status_code == 200:
                    st.success("✅ API Connected")
                else:
                    st.error(f"❌ Unexpected response: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ API Offline or Error: {e}")
    else:
        st.info("🔍 Click the button to check API status")


# noqa: E501
def show_ml_loggers():
    wandb_local_icon_path = PROJECT_ROOT / "assets" / "wandb_icon.png"
    mlflow_local_icon_path = PROJECT_ROOT / "assets" / "ml_flow_logo-white.png"
    tesnorboard_local_icon_path = PROJECT_ROOT / "assets" / "tensorboard-logo-social.png"

    wandb_icon_b64 = encode_image_path_to_base64(wandb_local_icon_path)
    mlflow_icon_b64 = encode_image_path_to_base64(mlflow_local_icon_path)
    tesnorboard_icon_b64 = encode_image_path_to_base64(tesnorboard_local_icon_path)

    ml_loggers_cfg: MLLoggersConfig = CONFIG.logging.ml_loggers

    st.markdown("## 📊 Track ML progress")
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <a href="{ml_loggers_cfg.wandb.tracking_uri}/{WANDB_ENTITY}/{ml_loggers_cfg.wandb.inference_project}"
               target="_blank"
               style="display: flex; align-items: center; justify-content: center;
               height: 75px; border: 1px solid #54555D; border-radius: 6px; padding: 4px;">
                <img src="data:image/png;base64,{wandb_icon_b64}"
                     style="max-height: 100%; max-width: 100%; object-fit: contain;" />
            </a>
            <a href="{ml_loggers_cfg.mlflow.tracking_uri}"
               target="_blank"
               style="display: flex; align-items: center; justify-content: center;
               height: 75px; border: 1px solid #54555D; border-radius: 6px; padding: 4px;">
                <img src="data:image/png;base64,{mlflow_icon_b64}"
                     style="max-height: 100%; max-width: 100%; object-fit: contain;" />
            </a>
            <a href="{ml_loggers_cfg.tensorboard.tracking_uri}"
               target="_blank"
               style="display: flex; align-items: center; justify-content: center;
               height: 75px; border: 1px solid #54555D; border-radius: 6px; padding: 4px;">
                <img src="data:image/png;base64,{tesnorboard_icon_b64}"
                     style="max-height: 100%; max-width: 100%; object-fit: contain;" />
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def configure_sidebar():
    with st.sidebar:
        st.image(PROJECT_ROOT / "assets" / "autovisionai_icon.png")
        show_ml_loggers()
        add_api_status()


def format_model_name(name):
    model_names_transform = {"unet": "U-Net", "fast_scnn": "Fast-SCNN", "mask_rcnn": "Mask R-CNN"}
    return model_names_transform[name]
