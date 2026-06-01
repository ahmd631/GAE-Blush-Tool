import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="GAE Blush ROI Tool",
    layout="wide"
)

st.title("GAE Blush ROI Annotation Tool")

st.write(
    "Select a case image from the repository, draw the blush ROI, save reader annotations, "
    "and compare interobserver agreement using Dice and IoU."
)

# --------------------------------------------------
# CREATE FOLDERS
# --------------------------------------------------

Path("annotations").mkdir(exist_ok=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

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
    min_value=1,
    max_value=10,
    value=3
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def find_images_in_main_folder():
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]

    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(".").glob(ext))

    image_files = sorted(image_files)

    # Avoid accidentally showing generated mask images if any exist
    image_files = [
        f for f in image_files
        if "_mask" not in f.stem.lower()
    ]

    return image_files


def resize_image_for_canvas(image, max_width=850):
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

    red = data[:, :, 0]
    green = data[:, :, 1]
    blue = data[:, :, 2]
    alpha = data[:, :, 3]

    # Detect red drawing pixels only
    mask = (
        (red > 120) &
        (green < 120) &
        (blue < 120) &
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


def create_overlay(image, mask1, mask2):
    image_array = np.array(image.convert("RGB"))

    if mask1.shape != image_array.shape[:2]:
        mask1 = np.array(
            Image.fromarray(mask1.astype(np.uint8) * 255).resize(
                (image_array.shape[1], image_array.shape[0])
            )
        ) > 0

    if mask2.shape != image_array.shape[:2]:
        mask2 = np.array(
            Image.fromarray(mask2.astype(np.uint8) * 255).resize(
                (image_array.shape[1], image_array.shape[0])
            )
        ) > 0

    overlay = np.zeros_like(image_array)

    intersection = np.logical_and(mask1, mask2)

    overlay[mask1] = [255, 0, 0]
    overlay[mask2] = [0, 255, 0]
    overlay[intersection] = [255, 255, 0]

    blended = image_array.copy()

    alpha = 0.45
    mask_any = np.logical_or(mask1, mask2)

    blended[mask_any] = (
        (1 - alpha) * image_array[mask_any] +
        alpha * overlay[mask_any]
    ).astype(np.uint8)

    return Image.fromarray(blended)


# --------------------------------------------------
# GET IMAGE FILES FROM MAIN FOLDER
# --------------------------------------------------

image_files = find_images_in_main_folder()

if len(image_files) == 0:
    st.error(
        "No image files found in the main repository folder. "
        "Upload PNG/JPG/TIF images directly beside app.py."
    )
    st.stop()

# --------------------------------------------------
# MODE 1: ANNOTATE ROI
# --------------------------------------------------

if mode == "Annotate ROI":

    st.header("Draw ROI")

    selected_image = st.selectbox(
        "Choose case image",
        image_files,
        format_func=lambda x: x.name
    )

    image = Image.open(selected_image).convert("RGB")

    case_id = selected_image.stem.replace(" ", "_")

    canvas_image = resize_image_for_canvas(
        image,
        max_width=850
    )

    st.success(f"Loaded case: {case_id}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Image Preview")
        st.image(
            canvas_image,
            caption=f"Preview: {selected_image.name}",
            use_container_width=False
        )

        st.subheader("Draw ROI on Image")

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.35)",
            stroke_width=stroke_width,
            stroke_color="rgba(255, 0, 0, 1)",
            background_image=canvas_image,
            update_streamlit=True,
            height=canvas_image.height,
            width=canvas_image.width,
            drawing_mode=drawing_mode,
            key=f"canvas_{case_id}_{reader_initials}_{drawing_mode}"
        )

    with col2:
        st.subheader("Annotation Info")

        st.write(f"**Case ID:** {case_id}")
        st.write(f"**Reader:** {reader_initials}")
        st.write(f"**Original size:** {image.width} × {image.height}")
        st.write(f"**Displayed size:** {canvas_image.width} × {canvas_image.height}")

        if st.button("Save Annotation"):

            if reader_initials == "":
                st.error("Please enter reader initials.")

            else:
                mask = canvas_to_mask(canvas_result)

                if mask is None or mask.sum() == 0:
                    st.error("No ROI detected. Draw on the image first.")

                else:
                    outfile = Path("annotations") / f"{case_id}_{reader_initials}.npy"

                    np.save(outfile, mask)

                    st.success("Annotation saved!")
                    st.write(f"Saved file: `{outfile}`")
                    st.write(f"ROI area: **{int(mask.sum())} pixels**")

                    mask_png = Image.fromarray(
                        (mask.astype(np.uint8) * 255)
                    )

                    st.image(
                        mask_png,
                        caption="Saved Binary ROI Mask",
                        use_container_width=True
                    )

                    with open(outfile, "rb") as f:
                        st.download_button(
                            label="Download Annotation Mask (.npy)",
                            data=f,
                            file_name=f"{case_id}_{reader_initials}.npy",
                            mime="application/octet-stream"
                        )


# --------------------------------------------------
# MODE 2: COMPARE READERS
# --------------------------------------------------

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

            area1, area2, overlap, dice, iou = calculate_metrics(
                mask1,
                mask2
            )

            rows.append({
                "Case": case_id,
                f"Area_{reader1}": area1,
                f"Area_{reader2}": area2,
                "Overlap": overlap,
                "Dice": dice,
                "IoU": iou
            })

        if len(rows) == 0:
            st.error(
                "No matching annotation pairs found. "
                "Make sure both readers saved masks for the same case."
            )

        else:
            results = pd.DataFrame(rows)

            st.subheader("Results Table")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                label="Download Results CSV",
                data=results.to_csv(index=False),
                file_name="GAE_Blush_Reader_Comparison.csv",
                mime="text/csv"
            )

            st.subheader("Study Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Number of Cases", len(results))
            col2.metric("Mean Dice", f"{results['Dice'].mean():.3f}")
            col3.metric("Mean IoU", f"{results['IoU'].mean():.3f}")
            col4.metric("Mean Overlap", f"{results['Overlap'].mean():.1f}")

            # Dice plot
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.boxplot(results["Dice"])
            ax1.set_ylabel("Dice")
            ax1.set_title("Dice Coefficient")
            st.pyplot(fig1)

            # IoU plot
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.boxplot(results["IoU"])
            ax2.set_ylabel("IoU")
            ax2.set_title("IoU")
            st.pyplot(fig2)

            # Scatter plot
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

            # Bland-Altman
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

            # Overlay section
            st.subheader("Overlay Viewer")

            selected_overlay_image = st.selectbox(
                "Choose image for overlay",
                image_files,
                format_func=lambda x: x.name,
                key="overlay_select"
            )

            overlay_case_id = selected_overlay_image.stem.replace(" ", "_")

            mask1_path = Path("annotations") / f"{overlay_case_id}_{reader1}.npy"
            mask2_path = Path("annotations") / f"{overlay_case_id}_{reader2}.npy"

            if mask1_path.exists() and mask2_path.exists():

                overlay_image = Image.open(selected_overlay_image).convert("RGB")

                mask1 = np.load(mask1_path)
                mask2 = np.load(mask2_path)

                overlay = create_overlay(
                    resize_image_for_canvas(overlay_image, max_width=850),
                    mask1,
                    mask2
                )

                st.write(
                    f"Red = {reader1}, Green = {reader2}, Yellow = overlap"
                )

                st.image(
                    overlay,
                    use_container_width=True
                )

            else:
                st.warning(
                    "No matching masks found for this selected overlay image."
                )
