import streamlit as st

st.set_page_config(
    page_title="GAE Blush Tool",
    layout="wide"
)

st.title("GAE Blush Annotation Platform")

st.markdown("""
### Welcome

Use the sidebar to navigate:

- Annotate → Draw ROI and save annotations
- Analysis → Calculate Dice, IoU, overlays, and dashboards
""")
