import streamlit as st

st.title("GAE Blush Annotation Tool")

uploaded_file = st.file_uploader(
    "Upload angiogram",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    st.success("Image uploaded!")
