import streamlit as st

# Define the pages
main_page = st.Page("pages/welcome.py", title="Welcome", icon="👋")
inference_page = st.Page("pages/inference.py", title="Inference", icon="🔍")
trainging_page = st.Page("pages/training.py", title="Training", icon="🎯")

# Set up navigation
pg = st.navigation([main_page, inference_page, trainging_page])

pg.run()
