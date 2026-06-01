import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="GAE Blush ROI Tool",
    layout="wide"
)

st.title("GAE Blush ROI Annotation Tool")

Path("annotations").mkdir(exist_ok=True)
Path("uploaded_images").mkdir(exist_ok=True)

st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Choose mode",
    ["Annotate ROI", "Compare Readers"]
)

reader_initials = st.sidebar.text_input(
    "Reader initials",
    value="AH"
).strip().upper()

drawing_mode = st.sidebar.selectbox(
    "Drawing tool",
    ["freedraw", "polygon", "rect", "circle"]
)

stroke_width = st.sidebar.slider(
    "Stroke width",
    1,
    10,
    3
)


def resize_image_for_canvas(image, max_width=800):
    original_width, original_height = image.size

    if original_width > max_width:
        scale = max_width / original_width
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = image.resize((new_width, new_height))

    return image


def canvas_to_mask(canvas_result):
    if canvas_result.image_data is None:
        return None

    data = canvas_result.image_data

    # Detect red drawing pixels
    red = data[:, :, 0]
    green = data[:, :, 1]
    blue = data[:, :, 2]
    alpha = data[:, :, 3]

    mask = (
        (red > 150) &
        (green < 100) &
        (blue < 100) &
        (alpha > 0)
    )

    return mask


def calculate_metrics(mask1, mask2):
    area1 = int(mask1.sum())
    area2 = int(mask2.sum())

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    overlap = int(intersection.sum())

    dice = 0
    iou = 0

    if area1 + area2 > 0:
        dice = (2 * overlap) / (area1 + area2)

    if union.sum() > 0:
        iou = overlap / union.sum()

    return area1, area2, overlap, round(dice, 3), round(iou, 3)


if mode == "Annotate ROI":

    st.header("Draw ROI")

    uploaded_file = st.file_uploader(
        "Upload angiogram image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        case_id = Path(uploaded_file.name).stem.replace(" ", "_")

        original_path = Path("uploaded_images") / f"{case_id}.png"
        image.save(original_path)

        canvas_image = resize_image_for_canvas(image, max_width=800)

        st.success(f"Loaded case: {case_id}")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Draw ROI on image")

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color="rgba(255, 0, 0, 1)",
                background_image=canvas_image,
                update_streamlit=True,
                height=canvas_image.height,
                width=canvas_image.width,
                drawing_mode=drawing_mode,
                key=f"canvas_{case_id}_{reader_initials}",
            )

        with col2:
            st.subheader("Annotation Info")
            st.write(f"**Case ID:** {case_id}")
            st.write(f"**Reader:** {reader_initials}")
            st.write(f"**Displayed size:** {canvas_image.width} × {canvas_image.height}")

            if st.button("Save Annotation"):

                mask = canvas_to_mask(canvas_result)

                if mask is None or mask.sum() == 0:
                    st.error("No ROI detected. Draw on the image first.")

                else:
                    outfile = Path("annotations") / f"{case_id}_{reader_initials}.npy"
                    np.save(outfile, mask)

                    st.success("Annotation saved!")
                    st.write(f"Saved: `{outfile}`")
                    st.write(f"ROI area: **{int(mask.sum())} pixels**")

                    st.image(
                        mask.astype(np.uint8) * 255,
                        caption="Saved ROI Mask",
                        use_container_width=True
                    )

    else:
        st.info("Upload an angiogram image first.")


if mode == "Compare Readers":

    st.header("Compare Two Readers")

    reader1 = st.text_input(
        "Reader 1 initials",
        value="AH"
    ).strip().upper()

    reader2 = st.text_input(
        "Reader 2 initials",
        value="YE"
    ).strip().upper()

    if st.button("Run Comparison"):

        files_reader1 = sorted(
            Path("annotations").glob(f"*_{reader1}.npy")
        )

        rows = []

        for file1 in files_reader1:

            case_id = file1.stem.replace(f"_{reader1}", "")
            file2 = Path("annotations") / f"{case_id}_{reader2}.npy"

            if not file2.exists():
                continue

            mask1 = np.load(file1)
            mask2 = np.load(file2)

            area1, area2, overlap, dice, iou = calculate_metrics(mask1, mask2)

            rows.append({
                "Case": case_id,
                f"Area_{reader1}": area1,
                f"Area_{reader2}": area2,
                "Overlap": overlap,
                "Dice": dice,
                "IoU": iou
            })

        if len(rows) == 0:
            st.error("No matching annotation pairs found.")

        else:
            results = pd.DataFrame(rows)

            st.subheader("Results")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download CSV",
                data=results.to_csv(index=False),
                file_name="GAE_Blush_Comparison.csv",
                mime="text/csv"
            )

            st.subheader("Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Cases", len(results))
            col2.metric("Mean Dice", f"{results['Dice'].mean():.3f}")
            col3.metric("Mean IoU", f"{results['IoU'].mean():.3f}")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot(results["Dice"])
            ax.set_ylabel("Dice")
            ax.set_title("Dice Coefficient")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.boxplot(results["IoU"])
            ax2.set_ylabel("IoU")
            ax2.set_title("IoU")
            st.pyplot(fig2)
