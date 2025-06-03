"""Utility functions for AutoVisionAI UI."""

import threading
from typing import Callable, Optional

import requests
import streamlit as st
import websocket


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
        api_url = st.session_state.get("api_base_url", "http://localhost:8000")
        response = requests.options(f"{api_url}{endpoint}", timeout=2)
        return response.status_code in (200, 204, 405)
    except requests.RequestException:
        return False


def add_sidebar_api_status():
    """Add API status sidebar with non-blocking checks."""
    with st.sidebar:
        st.markdown("## 🏎️ AutoVisionAI")
        st.markdown("---")

        # Initialize API URL in session state if not exists
        if "api_base_url" not in st.session_state:
            st.session_state.api_base_url = "http://localhost:8000"

        st.subheader("⚙️ API Settings")
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_base_url,
            help="Base URL of the AutoVisionAI API",
            key="global_api_url",
        )
        st.session_state.api_base_url = api_url

        # Connection status with spinner for better UX
        st.markdown("**Connection Status:**")
        with st.spinner("Checking API connection..."):
            try:
                response = requests.get(f"{api_url}/docs", timeout=2)
                if response.status_code == 200:
                    st.success("✅ API Connected")
                else:
                    st.error("❌ API Not Responding")
            except requests.exceptions.RequestException:
                st.error("❌ API Offline")
