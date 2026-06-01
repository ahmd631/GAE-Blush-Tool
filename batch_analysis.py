from skimage.io import imread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

READER1 = "AH"
READER2 = "YE"

rows = []

files = sorted(
    Path("annotations").glob(
        f"*_{READER1}.npy"
    )
)

for file in files:

    case_id = (
        file.stem
        .replace(f"_{READER1}", "")
    )

    file1 = (
        f"annotations/{case_id}_{READER1}.npy"
    )

    file2 = (
        f"annotations/{case_id}_{READER2}.npy"
    )

    if not Path(file2).exists():
        continue

    mask1 = np.load(file1)
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

    if area1 + area2 > 0:
        dice = (
            2 * overlap
        ) / (
            area1 + area2
        )
    else:
        dice = 0

    if union.sum() > 0:
        iou = (
            overlap /
            union.sum()
        )
    else:
        iou = 0

    rows.append({
        "Case": case_id,
        "Area_AH": int(area1),
        "Area_YE": int(area2),
        "Overlap": int(overlap),
        "Dice": round(dice, 3),
        "IoU": round(iou, 3)
    })

results = pd.DataFrame(rows)

mean_dice = results["Dice"].mean()
sd_dice = results["Dice"].std()

mean_iou = results["IoU"].mean()
sd_iou = results["IoU"].std()

# --------------------------------
# DASHBOARD
# --------------------------------

fig = plt.figure(
    figsize=(18, 12)
)

# --------------------------------
# TABLE
# --------------------------------

ax1 = plt.subplot(2, 3, 1)

ax1.axis("off")

table = ax1.table(
    cellText=results.values,
    colLabels=results.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.6)

ax1.set_title(
    "Case Results"
)
# --------------------------------
# DICE
# --------------------------------

ax2 = plt.subplot(2, 3, 2)

ax2.boxplot(
    results["Dice"]
)

ax2.set_ylabel(
    "Dice"
)

ax2.set_title(
    f"Dice\nMean={mean_dice:.3f}"
)

# --------------------------------
# IOU
# --------------------------------

ax3 = plt.subplot(2, 3, 3)

ax3.boxplot(
    results["IoU"]
)

ax3.set_ylabel(
    "IoU"
)

ax3.set_title(
    f"IoU\nMean={mean_iou:.3f}"
)

# --------------------------------
# SCATTER
# --------------------------------

ax4 = plt.subplot(2, 3, 4)

ax4.scatter(
    results["Area_AH"],
    results["Area_YE"]
)

max_val = max(
    results["Area_AH"].max(),
    results["Area_YE"].max()
)

ax4.plot(
    [0, max_val],
    [0, max_val],
    "--"
)

ax4.set_xlabel(
    "AH Area"
)

ax4.set_ylabel(
    "YE Area"
)

ax4.set_title(
    "Reader Area Correlation"
)

# --------------------------------
# BLAND ALTMAN
# --------------------------------

ax5 = plt.subplot(2, 3, 5)

mean_area = (
    results["Area_AH"] +
    results["Area_YE"]
) / 2

diff_area = (
    results["Area_AH"] -
    results["Area_YE"]
)

bias = diff_area.mean()

sd = diff_area.std()

ax5.scatter(
    mean_area,
    diff_area
)

ax5.axhline(
    bias,
    linestyle="--"
)

ax5.axhline(
    bias + 1.96 * sd,
    linestyle=":"
)

ax5.axhline(
    bias - 1.96 * sd,
    linestyle=":"
)

ax5.set_xlabel(
    "Mean Area"
)

ax5.set_ylabel(
    "Difference"
)

ax5.set_title(
    "Bland-Altman"
)

# --------------------------------
# SUMMARY
# --------------------------------

ax6 = plt.subplot(2, 3, 6)

ax6.axis("off")

summary = (
    f"N Cases: {len(results)}\n\n"
    f"Mean Dice: {mean_dice:.3f}\n"
    f"SD Dice: {sd_dice:.3f}\n\n"
    f"Mean IoU: {mean_iou:.3f}\n"
    f"SD IoU: {sd_iou:.3f}"
)

ax6.text(
    0.05,
    0.9,
    summary,
    fontsize=14,
    va="top",
    bbox=dict(
        facecolor="white",
        alpha=0.9
    )
)

ax6.set_title(
    "Study Summary"
)

plt.suptitle(
    "GAE Blush Interobserver Variability Analysis",
    fontsize=20
)

plt.tight_layout()

plt.savefig(
    "GAE_Blush_Full_Dashboard.png",
    dpi=300,
    bbox_inches="tight"
)
# --------------------------------
# SEPARATE OVERLAY FIGURE PAGE
# --------------------------------

n_cases = len(results)

fig2, axes = plt.subplots(
    1,
    n_cases,
    figsize=(6 * n_cases, 6)
)

if n_cases == 1:
    axes = [axes]

for idx, ax in enumerate(axes):

    case_id = results.iloc[idx]["Case"]

    image = imread(
        f"{case_id}.png"
    )

    mask1 = np.load(
        f"annotations/{case_id}_AH.npy"
    )

    mask2 = np.load(
        f"annotations/{case_id}_YE.npy"
    )

    intersection = np.logical_and(
        mask1,
        mask2
    )

    overlay = np.zeros(
        (*mask1.shape, 3),
        dtype=np.uint8
    )

    overlay[mask1] = [255, 0, 0]
    overlay[mask2] = [0, 255, 0]
    overlay[intersection] = [255, 255, 0]

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.imshow(
        overlay,
        alpha=0.5
    )

    dice = results.iloc[idx]["Dice"]

    iou = results.iloc[idx]["IoU"]

    ax.set_title(
        f"{case_id}\nDice={dice:.3f}\nIoU={iou:.3f}"
    )

    ax.axis("off")

plt.figure(fig2.number)

plt.savefig(
    "GAE_Blush_Overlay_Page.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print(
    "\nSaved: GAE_Blush_Full_Dashboard.png"
)
#python batch_analysis.py
