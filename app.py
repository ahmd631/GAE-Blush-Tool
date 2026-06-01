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

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Choose mode",
    ["Annotate ROI", "Compare Readers"]
)

reader_initials = st.sidebar.text_input(
    "Reader initials",
    value="AH"
).strip().upper()

stroke_width = st.sidebar.slider(
    "Stroke width",
    min_value=1,
    max_value=15,
    value=4
)

drawing_mode = st.sidebar.selectbox(
    "Drawing tool",
    ["freedraw", "polygon", "rect", "circle"]
)

# -------------------------------
# FUNCTIONS
# -------------------------------

def find_images():
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        files.extend(Path(".").glob(ext))

    files = sorted(files)

    files = [
        f for f in files
        if "_mask" not in f.stem.lower()
    ]

    return files


def prepare_canvas_image(image, max_width=700):
    image = image.convert("RGBA")

    width, height = image.size

    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    return image


def canvas_to_mask(canvas_result):
    if canvas_result.image_data is None:
        return None

    data = canvas_result.image_data.astype(np.uint8)

    red = data[:, :, 0]
    green = data[:, :, 1]
    blue = data[:, :, 2]

    # Detect red ROI drawing only
    mask = (
        (red > 180) &
        (green < 120) &
        (blue < 120)
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


# -------------------------------
# LOAD IMAGES FROM MAIN REPO
# -------------------------------

image_files = find_images()

if len(image_files) == 0:
    st.error("No image files found. Put your PNG/JPG images directly beside app.py.")
    st.stop()

# -------------------------------
# ANNOTATE MODE
# -------------------------------

if mode == "Annotate ROI":

    st.header("Annotate ROI")

    selected_image = st.selectbox(
        "Choose case image",
        image_files,
        format_func=lambda x: x.name
    )

    case_id = selected_image.stem.replace(" ", "_")

    original_image = Image.open(selected_image)
    canvas_image = prepare_canvas_image(original_image, max_width=700)

    st.success(f"Loaded case: {case_id}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Draw ROI directly on the image")

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.35)",
            stroke_width=stroke_width,
            stroke_color="rgba(255, 0, 0, 1)",
            background_image=canvas_image,
            background_color="white",
            update_streamlit=True,
            height=canvas_image.height,
            width=canvas_image.width,
            drawing_mode=drawing_mode,
            display_toolbar=True,
            key=f"{case_id}_{reader_initials}_{drawing_mode}"
        )

    with col2:
        st.subheader("Annotation Info")

        st.write(f"**Case ID:** {case_id}")
        st.write(f"**Reader:** {reader_initials}")
        st.write(f"**Original size:** {original_image.width} × {original_image.height}")
        st.write(f"**Canvas size:** {canvas_image.width} × {canvas_image.height}")

        if st.button("Save Annotation"):

            if reader_initials == "":
                st.error("Enter reader initials first.")

            else:
                mask = canvas_to_mask(canvas_result)

                if mask is None or mask.sum() == 0:
                    st.error("No ROI detected. Draw in red on the image first.")

                else:
                    outfile = Path("annotations") / f"{case_id}_{reader_initials}.npy"
                    np.save(outfile, mask)

                    st.success("Annotation saved!")
                    st.write(f"Saved: `{outfile}`")
                    st.write(f"ROI area: **{int(mask.sum())} pixels**")

                    mask_img = Image.fromarray(
                        (mask.astype(np.uint8) * 255)
                    )

                    st.image(
                        mask_img,
                        caption="Saved ROI Mask",
                        use_column_width=True
                    )

                    with open(outfile, "rb") as f:
                        st.download_button(
                            "Download Mask",
                            data=f,
                            file_name=f"{case_id}_{reader_initials}.npy",
                            mime="application/octet-stream"
                        )

# -------------------------------
# COMPARE MODE
# -------------------------------

if mode == "Compare Readers":

    st.header("Compare Readers")

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

            st.subheader("Results Table")
            st.dataframe(results)

            st.download_button(
                "Download CSV",
                data=results.to_csv(index=False),
                file_name="GAE_Blush_Comparison.csv",
                mime="text/csv"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Cases", len(results))
            col2.metric("Mean Dice", f"{results['Dice'].mean():.3f}")
            col3.metric("Mean IoU", f"{results['IoU'].mean():.3f}")

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.boxplot(results["Dice"])
            ax1.set_ylabel("Dice")
            ax1.set_title("Dice Coefficient")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.boxplot(results["IoU"])
            ax2.set_ylabel("IoU")
            ax2.set_title("IoU")
            st.pyplot(fig2)
