import streamlit as st

# Define the pages
main_page = st.Page("pages/1_welcome.py", title="Welcome to AutoVisionAI!", icon="👋")
inference_page = st.Page("pages/2_inference.py", title="Inference", icon="🔍")
trainging_page = st.Page("pages/3_training.py", title="Training", icon="🎯")

# Set up navigation
pg = st.navigation([main_page, inference_page, trainging_page])

pg.run()
