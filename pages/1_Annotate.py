import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas_jsretry import st_canvas

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
    image = image.convert("RGB")

    case_id = Path(
        uploaded_file.name
    ).stem

    st.subheader(
        f"Case: {case_id}"
    )

    st.image(
        image,
        caption="Angiogram",
        use_container_width=True
    )

    canvas_result = st_canvas(
        fill_color="rgba(255,0,0,0.3)",
        stroke_width=3,
        stroke_color="#ff0000",
        background_color="#000000",
        update_streamlit=True,
        height=600,
        width=800,
        drawing_mode="freedraw",
        key=f"canvas_{case_id}"
    )

    if st.button("Save Annotation"):

        if canvas_result.image_data is None:
            st.error(
                "Draw ROI first."
            )

        else:

            mask = (
                canvas_result.image_data[:, :, 3]
                > 0
            )

            st.write(
                f"Area = {mask.sum()} pixels"
            )
