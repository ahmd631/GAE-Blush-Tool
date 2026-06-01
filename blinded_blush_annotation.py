import napari
import numpy as np
from pathlib import Path
from skimage.io import imread
from magicgui import magicgui

# ---------------------------------
# SELECT CASE
# ---------------------------------

from pathlib import Path

png_files = sorted(
    Path(".").glob("*.png")
)

print("\nAvailable Cases:\n")

for i, file in enumerate(png_files):
    print(f"{i+1}. {file.name}")

choice = int(
    input(
        "\nSelect case number: "
    )
)

IMAGE_FILE = str(
    png_files[choice - 1]
)

CASE_ID = Path(
    IMAGE_FILE
).stem

# ---------------------------------
# LOAD IMAGE
# ---------------------------------

image = imread(
    IMAGE_FILE
)
viewer = napari.Viewer(
    title=f"GAE Blush Annotation - {CASE_ID}"
)

viewer.add_image(
    image,
    name="Angiogram"
)

roi_layer = viewer.add_shapes(
    name="Blush ROI",
    shape_type="path",
    edge_color="red",
    edge_width=2,
)

# ---------------------------------
# SAVE BUTTON
# ---------------------------------

@magicgui(
    call_button="Save Annotation",
    reader_initials={"label": "Your Initials"}
)
def save_annotation(
    reader_initials=""
):

    if len(roi_layer.data) == 0:
        print("No ROI drawn")
        return

    mask = roi_layer.to_labels(
        labels_shape=image.shape[:2]
    ) > 0

    Path("annotations").mkdir(
        exist_ok=True
    )

    outfile = (
        f"annotations/"
        f"{CASE_ID}_{reader_initials}.npy"
    )

    np.save(outfile, mask)

    print("\nSaved:")
    print(outfile)

    print(
        f"Area = {mask.sum()} pixels"
    )

viewer.window.add_dock_widget(
    save_annotation,
    area="right"
)

napari.run()
#python blinded_blush_annotation.py