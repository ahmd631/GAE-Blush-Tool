import streamlit as st
import numpy as np
from pathlib import Path
import pandas as pd

st.set_page_config(layout="wide")

st.title("GAE Blush Analysis")

reader1 = st.text_input(
    "Reader 1",
    value="AH"
)

reader2 = st.text_input(
    "Reader 2",
    value="YE"
)

files = sorted(
    Path("annotations").glob(
        f"*_{reader1}.npy"
    )
)

rows = []

for file in files:

    case_id = (
        file.stem
        .replace(
            f"_{reader1}",
            ""
        )
    )

    file2 = (
        Path("annotations")
        /
        f"{case_id}_{reader2}.npy"
    )

    if not file2.exists():
        continue

    mask1 = np.load(file)
    mask2 = np.load(file2)

    area1 = mask1.sum()
    area2 = mask2.sum()

    intersection = np.logical_and(
        mask1,
        mask2
    )

    union = np.logical_or(
        mask1,
        mask2
    )

    overlap = intersection.sum()

    dice = (
        2 * overlap
    ) / (
        area1 + area2
    )

    iou = (
        overlap /
        union.sum()
    )

    rows.append({
        "Case": case_id,
        "Area_AH": int(area1),
        "Area_YE": int(area2),
        "Dice": round(
            dice,
            3
        ),
        "IoU": round(
            iou,
            3
        )
    })

results = pd.DataFrame(
    rows
)

st.dataframe(
    results,
    use_container_width=True
)

if len(results):

    st.metric(
        "Mean Dice",
        round(
            results["Dice"].mean(),
            3
        )
    )

    st.metric(
        "Mean IoU",
        round(
            results["IoU"].mean(),
            3
        )
    )
