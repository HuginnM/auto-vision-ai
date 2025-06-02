"""Welcome page for AutoVisionAI Streamlit UI."""

import requests
import streamlit as st


def show_welcome_page():
    """Show the welcome page with project information and instructions."""
    st.title("🚗 Welcome to AutoVisionAI")

    # Hero section
    st.markdown("""
    ### Production-Ready Car Segmentation Pipeline

    AutoVisionAI is a modern computer vision pipeline featuring state-of-the-art architectures
    for car segmentation, implemented with PyTorch Lightning and modern Python tooling.
    """)

    # Key features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🧠 **Modern Architectures**
        - **U-Net**: Classic encoder-decoder
        - **Fast-SCNN**: Optimized for real-time
        - **Mask R-CNN**: Instance segmentation
        """)

    with col2:
        st.markdown("""
        #### ⚡ **Production Ready**
        - PyTorch Lightning framework
        - MLflow & Weights & Biases integration
        - Dockerized deployment
        - CI/CD pipeline
        """)

    with col3:
        st.markdown("""
        #### 🛠️ **Modern Tooling**
        - UV package manager
        - Ruff linting & formatting
        - Pre-commit hooks
        - Comprehensive testing
        """)

    st.markdown("---")

    # Quick start guide
    st.header("🚀 Quick Start Guide")

    tab1, tab2, tab3 = st.tabs(["🔍 Inference", "🎯 Training", "⚙️ Setup"])

    with tab1:
        st.markdown("""
        ### Run Inference on Your Images

        1. **Navigate to the Inference page** using the sidebar
        2. **Choose input method:**
           - Upload an image file (PNG, JPG, JPEG)
           - Provide an image URL
        3. **Select a model** (UNet, Fast-SCNN, or Mask R-CNN)
        4. **Click "Run Inference"** to process your image
        5. **View results** including segmentation mask and metrics

        #### Supported Models:
        - **UNet**: Best for general segmentation tasks (~31M params)
        - **Fast-SCNN**: Optimized for speed (~1.1M params)
        - **Mask R-CNN**: Instance segmentation with detection (~44M params)
        """)

    with tab2:
        st.markdown("""
        ### Train Your Own Models

        1. **Navigate to the Training page** using the sidebar
        2. **Configure experiment settings:**
           - Experiment name
           - Model architecture
        3. **Set hyperparameters:**
           - Batch size, epochs, early stopping
           - Data augmentation options
        4. **Click "Start Training"** to begin
        5. **Monitor progress** with real-time updates:
           - Training loss graphs
           - Epoch progress
           - Training logs

        #### Training Features:
        - **Real-time monitoring** via WebSocket connections
        - **Interactive loss graphs** updated live
        - **Early stopping** to prevent overfitting
        - **Data augmentation** for better generalization
        """)

    with tab3:
        st.markdown("""
        ### Setup Instructions

        #### 1. Start the API Server
        ```bash
        # Start the AutoVisionAI API server
        python -m autovisionai.api.main
        ```

        #### 2. Launch the UI
        ```bash
        # Start the Streamlit UI
        python scripts/run_ui.py
        ```

        #### 3. Configure API Connection
        - Check the **API Settings** in the sidebar
        - Default API URL: `http://localhost:8000`
        - Ensure the connection status shows API Connected

        #### 4. Verify Setup
        - API server should be running on port 8000
        - UI should be accessible at `http://localhost:8501`
        - Both inference and training endpoints should be available
        """)

    st.markdown("---")

    # System status
    st.header("📊 System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔗 API Connection")
        api_url = st.session_state.get("api_base_url", "http://localhost:8000")

        try:
            response = requests.get(f"{api_url}/docs", timeout=2)
            if response.status_code == 200:
                st.success(f"✅ API server is running at {api_url}")

                # Test endpoints
                try:
                    # Test inference endpoint
                    requests.get(f"{api_url}/inference/", timeout=2)
                    requests.get(f"{api_url}/train/", timeout=2)

                    st.info("📡 **Available Endpoints:**")
                    st.write("- 🔍 Inference: `/inference/`")
                    st.write("- 🎯 Training: `/train/`")
                    st.write("- 🌐 WebSocket: `/train/ws/`")

                except requests.exceptions.RequestException:
                    st.warning("Some endpoints may not be fully available")

            else:
                st.error(f"API server responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API server at {api_url}")
            st.info("**Troubleshooting:**")
            st.write("1. Make sure the API server is running")
            st.write("2. Check the API URL in sidebar settings")
            st.write("3. Verify port 8000 is not blocked")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")

    with col2:
        st.subheader("🎛️ UI Information")
        st.info("**Current Session:**")
        st.write("- **UI Port:** 8501")
        st.write(f"- **API URL:** {api_url}")
        st.write(f"- **Session State:** {len(st.session_state)} variables")

        if st.button("🔄 Refresh Status", key="refresh_status"):
            st.rerun()

    st.markdown("---")

    # Footer
    st.markdown("""
    ### 📚 Additional Resources

    - **GitHub Repository**: [AutoVisionAI](https://github.com/huginnm/auto-vision-ai)
    - **Documentation**: Check the README.md files in each component
    - **API Documentation**: Visit `/docs` endpoint when API is running
    - **Model Weights**: Available through MLflow tracking

    ---
    **Made using PyTorch Lightning, FastAPI, and Streamlit**
    """)

    # Quick actions
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Go to Inference", type="primary", use_container_width=True, key="goto_inference"):
            st.session_state.current_page = "Inference"
            st.rerun()

    with col2:
        if st.button("🎯 Go to Training", type="primary", use_container_width=True, key="goto_training"):
            st.session_state.current_page = "Training"
            st.rerun()

    with col3:
        if st.button("📖 View API Docs", use_container_width=True, key="view_docs"):
            api_url = st.session_state.get("api_base_url", "http://localhost:8000")
            st.markdown(f"[Open API Documentation]({api_url}/docs)")
            st.info(f"API docs should open at: {api_url}/docs")
