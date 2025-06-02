"""Main Streamlit application for AutoVisionAI."""

import requests
import streamlit as st

from autovisionai.ui.pages.inference_page import show_inference_page
from autovisionai.ui.pages.training_page import show_training_page
from autovisionai.ui.pages.welcome_page import show_welcome_page


def main():
    """Main application entry point."""
    st.set_page_config(page_title="AutoVisionAI", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

    # Sidebar navigation
    st.sidebar.title("AutoVisionAI")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox("Select Page", ["Welcome", "Inference", "Training"], index=0)

    st.sidebar.markdown("---")

    # API configuration in sidebar
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"

    with st.sidebar:
        st.subheader("⚙️ API Settings")
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_base_url,
            help="Base URL of the AutoVisionAI API",
            key="global_api_url",
        )
        st.session_state.api_base_url = api_url

        # Connection status
        st.markdown("**Connection Status:**")
        try:
            response = requests.get(f"{api_url}/docs", timeout=2)
            if response.status_code == 200:
                st.success("API Connected")
            else:
                st.error("API Not Responding")
        except requests.exceptions.RequestException:
            st.error("API Offline")

    # Main content
    if page == "Welcome":
        show_welcome_page()
    elif page == "Inference":
        show_inference_page()
    elif page == "Training":
        show_training_page()


if __name__ == "__main__":
    main()
