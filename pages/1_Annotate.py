import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

st.title("GAE Blush Annotation")

reader = st.text_input(
    "Reader Initials",
    placeholder="AH"
)

uploaded_file = st.file_uploader(
    "Upload Angiogram",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Angiogram",
        use_container_width=True
    )

    st.success(
        f"Ready for annotation ({reader})"
    )

    st.info(
        "Next step: ROI drawing canvas"
    )
