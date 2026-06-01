
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="GAE Blush ROI Tool",
    layout="wide"
)

st.title("GAE Blush ROI Annotation Tool")
st.write(
    "Upload an angiogram image, draw the blush ROI, save reader annotations, "
    "and compare interobserver agreement using Dice and IoU."
)

# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

Path("annotations").mkdir(exist_ok=True)
Path("uploaded_images").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Choose mode",
    [
        "1. Annotate ROI",
        "2. Compare Readers"
    ]
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
    min_value=1,
    max_value=10,
    value=3
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def save_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    case_id = Path(uploaded_file.name).stem.replace(" ", "_")
    image_path = Path("uploaded_images") / f"{case_id}.png"
    image.save(image_path)
    return case_id, image, image_path


def canvas_to_mask(canvas_result, image_shape):
    """
    Converts the canvas RGBA drawing layer into a binary mask.
    """
    if canvas_result.image_data is None:
        return None

    canvas_data = canvas_result.image_data

    # Alpha channel: drawn areas have alpha > 0
    alpha_channel = canvas_data[:, :, 3]

    mask = alpha_channel > 0

    # Make sure mask shape matches image H x W
    if mask.shape != image_shape[:2]:
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (image_shape[1], image_shape[0])
            )
        ) > 0

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

    return {
        "Area Reader 1": area1,
        "Area Reader 2": area2,
        "Overlap": overlap,
        "Dice": round(dice, 3),
        "IoU": round(iou, 3)
    }


def create_overlay(image, mask1, mask2):
    image_array = np.array(image.convert("RGB"))

    overlay = np.zeros_like(image_array)

    intersection = np.logical_and(mask1, mask2)

    overlay[mask1] = [255, 0, 0]      # Reader 1 red
    overlay[mask2] = [0, 255, 0]      # Reader 2 green
    overlay[intersection] = [255, 255, 0]  # overlap yellow

    blended = image_array.copy()
    alpha = 0.45

    mask_any = np.logical_or(mask1, mask2)
    blended[mask_any] = (
        (1 - alpha) * image_array[mask_any] +
        alpha * overlay[mask_any]
    ).astype(np.uint8)

    return Image.fromarray(blended)


# --------------------------------------------------
# MODE 1: ANNOTATION
# --------------------------------------------------

if mode == "1. Annotate ROI":

    st.header("Step 1: Upload Image and Draw ROI")

    uploaded_file = st.file_uploader(
        "Upload angiogram PNG/JPG",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        case_id, image, image_path = save_uploaded_image(uploaded_file)

        st.success(f"Loaded case: {case_id}")

        image_array = np.array(image)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Draw ROI")

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color="red",
                background_image=image,
                update_streamlit=True,
                height=image.height,
                width=image.width,
                drawing_mode=drawing_mode,
                key=f"canvas_{case_id}_{reader_initials}"
            )

        with col2:
            st.subheader("Annotation Info")
            st.write(f"**Case ID:** {case_id}")
            st.write(f"**Reader:** {reader_initials}")
            st.write(f"**Image size:** {image.width} × {image.height}")

            if st.button("Save Annotation"):

                if reader_initials == "":
                    st.error("Please enter reader initials.")

                else:
                    mask = canvas_to_mask(
                        canvas_result,
                        image_array.shape
                    )

                    if mask is None or mask.sum() == 0:
                        st.error("No ROI detected. Please draw on the image first.")

                    else:
                        outfile = Path("annotations") / f"{case_id}_{reader_initials}.npy"
                        np.save(outfile, mask)

                        st.success("Annotation saved!")
                        st.write(f"Saved file: `{outfile}`")
                        st.write(f"ROI area: **{int(mask.sum())} pixels**")

                        mask_png = Image.fromarray(
                            (mask.astype(np.uint8) * 255)
                        )

                        mask_png_path = Path("annotations") / f"{case_id}_{reader_initials}_mask.png"
                        mask_png.save(mask_png_path)

                        st.image(
                            mask_png,
                            caption="Saved Binary ROI Mask",
                            use_container_width=True
                        )

    else:
        st.info("Upload an image to begin annotation.")


# --------------------------------------------------
# MODE 2: COMPARE READERS
# --------------------------------------------------

if mode == "2. Compare Readers":

    st.header("Step 2: Compare Two Readers")

    reader1 = st.text_input(
        "Reader 1 initials",
        value="AH"
    ).strip().upper()

    reader2 = st.text_input(
        "Reader 2 initials",
        value="YE"
    ).strip().upper()

    uploaded_compare_image = st.file_uploader(
        "Upload the same angiogram image for overlay",
        type=["png", "jpg", "jpeg"],
        key="compare_image"
    )

    if st.button("Run Comparison"):

        if reader1 == "" or reader2 == "":
            st.error("Please enter both reader initials.")

        else:
            files_reader1 = sorted(
                Path("annotations").glob(f"*_{reader1}.npy")
            )

            rows = []
            overlays = []

            for file1 in files_reader1:

                case_id = file1.stem.replace(f"_{reader1}", "")
                file2 = Path("annotations") / f"{case_id}_{reader2}.npy"

                if not file2.exists():
                    continue

                mask1 = np.load(file1)
                mask2 = np.load(file2)

                metrics = calculate_metrics(mask1, mask2)

                row = {
                    "Case": case_id,
                    f"Area_{reader1}": metrics["Area Reader 1"],
                    f"Area_{reader2}": metrics["Area Reader 2"],
                    "Overlap": metrics["Overlap"],
                    "Dice": metrics["Dice"],
                    "IoU": metrics["IoU"]
                }

                rows.append(row)

            if len(rows) == 0:
                st.error(
                    "No matching annotation pairs found. "
                    "Make sure both readers saved masks for the same case."
                )

            else:
                results = pd.DataFrame(rows)

                st.subheader("Results Table")
                st.dataframe(results, use_container_width=True)

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="GAE_Blush_Reader_Comparison.csv",
                    mime="text/csv"
                )

                mean_dice = results["Dice"].mean()
                sd_dice = results["Dice"].std()

                mean_iou = results["IoU"].mean()
                sd_iou = results["IoU"].std()

                st.subheader("Study Summary")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Number of Cases", len(results))
                col2.metric("Mean Dice", f"{mean_dice:.3f}")
                col3.metric("Mean IoU", f"{mean_iou:.3f}")
                col4.metric("Mean Overlap", f"{results['Overlap'].mean():.1f}")

                # ------------------------------
                # Dice Plot
                # ------------------------------

                st.subheader("Dice Distribution")

                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.boxplot(results["Dice"])
                ax1.set_ylabel("Dice")
                ax1.set_title(f"Dice Coefficient\nMean = {mean_dice:.3f}")
                st.pyplot(fig1)

                # ------------------------------
                # IoU Plot
                # ------------------------------

                st.subheader("IoU Distribution")

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.boxplot(results["IoU"])
                ax2.set_ylabel("IoU")
                ax2.set_title(f"IoU\nMean = {mean_iou:.3f}")
                st.pyplot(fig2)

                # ------------------------------
                # Area Scatter
                # ------------------------------

                st.subheader("Reader Area Correlation")

                area1_col = f"Area_{reader1}"
                area2_col = f"Area_{reader2}"

                fig3, ax3 = plt.subplots(figsize=(6, 5))
                ax3.scatter(results[area1_col], results[area2_col])

                max_val = max(
                    results[area1_col].max(),
                    results[area2_col].max()
                )

                ax3.plot([0, max_val], [0, max_val], "--")
                ax3.set_xlabel(f"{reader1} Area")
                ax3.set_ylabel(f"{reader2} Area")
                ax3.set_title("Reader Area Correlation")
                st.pyplot(fig3)

                # ------------------------------
                # Bland Altman
                # ------------------------------

                st.subheader("Bland-Altman Plot")

                mean_area = (
                    results[area1_col] +
                    results[area2_col]
                ) / 2

                diff_area = (
                    results[area1_col] -
                    results[area2_col]
                )

                bias = diff_area.mean()
                sd = diff_area.std()

                fig4, ax4 = plt.subplots(figsize=(6, 5))
                ax4.scatter(mean_area, diff_area)
                ax4.axhline(bias, linestyle="--")
                ax4.axhline(bias + 1.96 * sd, linestyle=":")
                ax4.axhline(bias - 1.96 * sd, linestyle=":")
                ax4.set_xlabel("Mean Area")
                ax4.set_ylabel("Difference")
                ax4.set_title("Bland-Altman Plot")
                st.pyplot(fig4)

                # ------------------------------
                # Overlay for one uploaded image
                # ------------------------------

                if uploaded_compare_image is not None:

                    compare_case_id, compare_image, _ = save_uploaded_image(
                        uploaded_compare_image
                    )

                    mask1_path = Path("annotations") / f"{compare_case_id}_{reader1}.npy"
                    mask2_path = Path("annotations") / f"{compare_case_id}_{reader2}.npy"

                    if mask1_path.exists() and mask2_path.exists():

                        mask1 = np.load(mask1_path)
                        mask2 = np.load(mask2_path)

                        overlay_image = create_overlay(
                            compare_image,
                            mask1,
                            mask2
                        )

                        st.subheader("Reader Overlay")
                        st.write(
                            f"Red = {reader1}, Green = {reader2}, Yellow = overlap"
                        )

                        st.image(
                            overlay_image,
                            use_container_width=True
                        )

                    else:
                        st.warning(
                            "No matching masks found for this uploaded comparison image."
                        )
