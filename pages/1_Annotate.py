import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

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

    case_id = Path(
        uploaded_file.name
    ).stem

    st.subheader(
        f"Case: {case_id}"
    )

    canvas_result = st_canvas(
        fill_color="rgba(255,0,0,0.3)",
        stroke_width=3,
        stroke_color="#ff0000",
        background_image=image,
        update_streamlit=True,
        height=image.size[1],
        width=image.size[0],
        drawing_mode="freedraw",
        key="canvas"
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

            Path(
                "annotations"
            ).mkdir(
                exist_ok=True
            )

            outfile = (
                f"annotations/"
                f"{case_id}_{reader}.npy"
            )

            np.save(
                outfile,
                mask
            )

            st.success(
                f"Saved: {outfile}"
            )

            st.write(
                f"Area = {mask.sum()} pixels"
            )
