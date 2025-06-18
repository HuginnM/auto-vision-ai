"""Welcome page for AutoVisionAI Streamlit UI."""

import streamlit as st

from autovisionai.core.configs import CONFIG
from autovisionai.ui.utils import check_api_endpoint, configure_sidebar


def main():
    st.set_page_config(page_title="AutoVisionAI", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded")

    # API configuration in sidebar (shared across all pages)
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = CONFIG.app.api_base_url

    configure_sidebar()

    # Main page content
    st.title("👋 Welcome to AutoVisionAI")
    st.markdown("**Advanced Car Segmentation with Deep Learning**")

    # Hero section with project description
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        AutoVisionAI is a comprehensive platform for car segmentation using state-of-the-art deep learning models.
        This application provides an intuitive interface for running inference on images and training custom models.

        ### ✨ Key Features
        - **🔍 Real-time Inference**: Upload images or provide URLs for instant car segmentation
        - **🎯 Model Training**: Train custom models with configurable hyperparameters
        - **📊 Progress Monitoring**: Real-time training progress with interactive visualizations
        - **🤖 Multiple Architectures**: Support for U-Net, Fast-SCNN, and Mask R-CNN models
        - **🚀 REST API**: Full-featured API for programmatic access
        """
        )

    with col2:
        st.info(
            """
        **Quick Start:**
        1. 🔍 Go to **Inference** to test models
        2. 🎯 Use **Training** to create custom models
        3. ⚙️ Configure API settings in the sidebar
        """
        )

    # Quick start guide with tabs
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")

    tab1, tab2, tab3 = st.tabs(["🔍 Running Inference", "🎯 Training Models", "🔧 API Setup"])

    with tab1:
        st.markdown(
            """
        ### Running Inference

        1. **Navigate to the Inference page** using the menu
        2. **Select a model** (U-Net, Fast-SCNN, or Mask R-CNN)
        3. **Upload an image** or provide an image URL
        4. **Click "Run Inference"** to get segmentation results
        5. **View results** with mask overlay visualization
        6. **Download results** as reports or images

        #### Supported Image Formats
        - PNG, JPG, JPEG
        - Maximum file size: 10MB
        - Recommended resolution: 512x512 or higher
        """
        )

    with tab2:
        st.markdown(
            """
        ### Training Custom Models

        1. **Navigate to the Training page** using the menu
        2. **Configure hyperparameters**:
        - Model architecture (U-Net, Fast-SCNN, Mask R-CNN)
        - Learning rate, batch size, epochs
        - Dataset path and training split
        3. **Set advanced options** (optimizer, scheduler, etc.)
        4. **Start training** and monitor progress in real-time
        5. **View training curves** and logs
        6. **Download training logs** when complete

        #### Dataset Requirements
        - Images and corresponding segmentation masks
        - Organized in standard format (images/ and masks/ folders)
        - Supported formats: PNG, JPG for images; PNG for masks
        """
        )

    with tab3:
        st.markdown(
            f"""
        ### API Setup

        1. **Start the API server**:
        ```bash
        cd AutoVisionAI
        python -m uvicorn src.autovisionai.api.main:app --reload
        ```

        2. **Configure API URL** in the sidebar (default: {CONFIG.app.api_base_url})

        3. **Test connection** - status shown in sidebar

        4. **Access API documentation** at {CONFIG.app.api_base_url}/docs

        #### Available Endpoints
        - `POST /inference/` - Run inference on images
        - `POST /train` - Start training jobs
        - `GET /models/` - List available models
        - `GET /train/ws/{{experiment_name}}` - Monitor training progress with WebSocket
        - `GET /health` - Check API health
        """
        )

    # System status section
    st.markdown("---")
    st.subheader("🖥️ System Status")

    # Button at the top
    check_endpoints = st.button("🔄 Check Endpoints", type="secondary", use_container_width=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**API Endpoints Status**")

        if check_endpoints:
            # Show actual status when button is pressed
            endpoints = [
                ("/docs", "API Documentation"),
                ("/inference", "Inference Endpoint"),
                ("/train", "Training Endpoint"),
            ]

            for endpoint, description in endpoints:
                with st.spinner(f"Checking {endpoint} connection..."):
                    if check_api_endpoint(endpoint):
                        st.success(f"✅ {description}")
                    else:
                        st.error(f"❌ {description}")
        else:
            # Placeholder content before button is pressed
            st.markdown("""
            📋 **Available Endpoints:**
            - 📚 API Documentation (`/docs`)
            - 🔍 Inference Endpoint (`/inference`)
            - 🎯 Training Endpoint (`/train`)

            Click the button above to test connectivity to each endpoint.
            """)

    with col2:
        st.markdown("**System Information**")

        # Get API URL from session state
        api_url = st.session_state.get("api_base_url", CONFIG.app.api_base_url)

        st.write(f"**API URL:** {api_url}")
        st.write("**UI Framework:** Streamlit")
        st.write("**Backend:** FastAPI")
        st.write("**ML Framework:** PyTorch")

    # Documentation links
    st.markdown("---")
    st.subheader("📖 Documentation & Resources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        **📚 Documentation**
        - [Project README](https://github.com/HuginnM/auto-vision-ai/blob/main/README.md)
        - [API Documentation]({CONFIG.app.api_base_url}/docs)
        """
        )

    with col2:
        st.markdown(
            """
        **🔗 Quick Links**
        - [🔍 Inference Page](Inference)
        - [🎯 Training Page](Training)
        - [⚙️ Settings](#)
        """
        )

    with col3:
        st.markdown(
            """
        **🆘 Support**
        - [GitHub Issues](https://github.com/your-repo/issues)
        - [Discussions](https://github.com/your-repo/discussions)
        - [FAQ](https://github.com/your-repo/wiki)
        """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            AutoVisionAI • Built with ❤️ using Streamlit and FastAPI
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
