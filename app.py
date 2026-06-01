import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.draw import polygon

st.set_page_config(
    page_title="GAE Blush ROI Tool",
    layout="wide"
)

st.title("GAE Blush ROI Annotation Tool")
st.write(
    "Click around the blush region to create a polygon ROI. "
    "Save masks for two readers, then compare Dice and IoU."
)

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

max_width = st.sidebar.slider(
    "Display image width",
    min_value=400,
    max_value=1000,
    value=700,
    step=50
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


def resize_image(image, max_width=700):
    image = image.convert("RGB")
    width, height = image.size

    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def draw_points_and_polygon(image, points):
    display = image.copy()
    draw = ImageDraw.Draw(display)

    if len(points) > 1:
        draw.line(points, fill="red", width=3)

    if len(points) > 2:
        draw.line([points[-1], points[0]], fill="red", width=3)

    for i, point in enumerate(points):
        x, y = point
        r = 5
        draw.ellipse((x-r, y-r, x+r, y+r), fill="yellow", outline="red")
        draw.text((x+7, y+7), str(i+1), fill="red")

    return display


def points_to_mask(points, shape):
    if len(points) < 3:
        return None

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    rr, cc = polygon(ys, xs, shape=shape)

    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True

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


def overlay_masks(image, mask1, mask2):
    image_array = np.array(image.convert("RGB"))

    overlay = np.zeros_like(image_array)
    intersection = np.logical_and(mask1, mask2)

    overlay[mask1] = [255, 0, 0]
    overlay[mask2] = [0, 255, 0]
    overlay[intersection] = [255, 255, 0]

    blended = image_array.copy()
    mask_any = np.logical_or(mask1, mask2)

    blended[mask_any] = (
        0.55 * image_array[mask_any] +
        0.45 * overlay[mask_any]
    ).astype(np.uint8)

    return Image.fromarray(blended)


# -------------------------------
# LOAD IMAGES
# -------------------------------

image_files = find_images()

if len(image_files) == 0:
    st.error("No images found. Put your PNG/JPG images directly beside app.py.")
    st.stop()

# -------------------------------
# ANNOTATE MODE
# -------------------------------

if mode == "Annotate ROI":

    selected_image = st.selectbox(
        "Choose case image",
        image_files,
        format_func=lambda x: x.name
    )

    case_id = selected_image.stem.replace(" ", "_")

    if "active_case" not in st.session_state:
        st.session_state.active_case = case_id

    if st.session_state.active_case != case_id:
        st.session_state.active_case = case_id
        st.session_state.points = []

    if "points" not in st.session_state:
        st.session_state.points = []

    original_image = Image.open(selected_image).convert("RGB")
    display_image = resize_image(original_image, max_width=max_width)

    annotated_display = draw_points_and_polygon(
        display_image,
        st.session_state.points
    )

    st.success(f"Loaded case: {case_id}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Click around the ROI border")

        value = streamlit_image_coordinates(
            annotated_display,
            key=f"coords_{case_id}_{reader_initials}_{len(st.session_state.points)}"
        )

        if value is not None:
            point = (int(value["x"]), int(value["y"]))

            if (
                len(st.session_state.points) == 0
                or point != st.session_state.points[-1]
            ):
                st.session_state.points.append(point)
                st.rerun()

    with col2:
        st.subheader("Annotation Info")
        st.write(f"**Case ID:** {case_id}")
        st.write(f"**Reader:** {reader_initials}")
        st.write(f"**Original size:** {original_image.width} × {original_image.height}")
        st.write(f"**Display size:** {display_image.width} × {display_image.height}")
        st.write(f"**Points clicked:** {len(st.session_state.points)}")

        if st.button("Undo last point"):
            if len(st.session_state.points) > 0:
                st.session_state.points.pop()
                st.rerun()

        if st.button("Clear ROI"):
            st.session_state.points = []
            st.rerun()

        if st.button("Save Annotation"):

            if reader_initials == "":
                st.error("Enter reader initials first.")

            elif len(st.session_state.points) < 3:
                st.error("Click at least 3 points to make a polygon ROI.")

            else:
                mask = points_to_mask(
                    st.session_state.points,
                    shape=(display_image.height, display_image.width)
                )

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

            st.subheader("Overlay Viewer")

            selected_overlay_image = st.selectbox(
                "Choose image for overlay",
                image_files,
                format_func=lambda x: x.name,
                key="overlay_case"
            )

            overlay_case_id = selected_overlay_image.stem.replace(" ", "_")

            mask1_path = Path("annotations") / f"{overlay_case_id}_{reader1}.npy"
            mask2_path = Path("annotations") / f"{overlay_case_id}_{reader2}.npy"

            if mask1_path.exists() and mask2_path.exists():
                original_image = Image.open(selected_overlay_image).convert("RGB")
                display_image = resize_image(original_image, max_width=max_width)

                mask1 = np.load(mask1_path)
                mask2 = np.load(mask2_path)

                overlay = overlay_masks(display_image, mask1, mask2)

                st.write(f"Red = {reader1}, Green = {reader2}, Yellow = overlap")
                st.image(overlay, use_column_width=True)

            else:
                st.warning("No matching masks found for this selected image.")
