import logging

import streamlit as st

from autovisionai.core.configs import ENV_MODE

logger = logging.getLogger(__name__)

logger.info(f"Starting AutoVisionAI UI in {ENV_MODE} mode")

# Define the pages
main_page = st.Page("pages/welcome.py", title="Welcome", icon="👋")
inference_page = st.Page("pages/inference.py", title="Inference", icon="🔍")
trainging_page = st.Page("pages/training.py", title="Training", icon="🎯")

# Set up navigation
pg = st.navigation([main_page, inference_page, trainging_page])

pg.run()
