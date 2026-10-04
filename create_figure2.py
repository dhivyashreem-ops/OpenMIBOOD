from pathlib import Path
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\dhivy\U-RELIABILITY-MEDICAL\OpenMIBOOD"
)

JSON_FILE = ROOT / "data" / "raw" / "MIDOG" / "MIDOGpp.json"

OUT_DIR = ROOT / "paper_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "Figure_2_Actual_Overlap_Cases.png"


# ============================================================
# VERIFIED OVERLAP CASES
# ============================================================

CASES = [
    {
        "source": ROOT / "data" / "raw" / "MIDOG" / "domain1a" / "024.tiff",
        "parent_id": 522,
        "overlap_id": 491,
        "case": "A",
    },
    {
        "source": ROOT / "data" / "raw" / "MIDOG" / "domain1a" / "034.tiff",
        "parent_id": 772,
        "overlap_id": 773,
        "case": "B",
    },
    {
        "source": ROOT / "data" / "raw" / "MIDOG" / "domain1a" / "039.tiff",
        "parent_id": 886,
        "overlap_id": 885,
        "case": "C",
    },
]


# ============================================================
# BACKGROUND-CROP GENERATION RULE
# ============================================================

SHIFT_X = 100
SHIFT_Y = 0


# ============================================================
# LOAD COCO JSON
# ============================================================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# HELPERS
# ============================================================

def get_annotation(annotation_id):

    for ann in data["annotations"]:

        if ann["id"] == annotation_id:
            return ann

    raise RuntimeError(
        f"Annotation {annotation_id} not found."
    )


def xywh_to_xyxy(bbox):

    # MIDOG++ annotations in this project are stored as:
    # [x1, y1, x2, y2]
    #
    # Do NOT interpret the last two values as width/height.

    x1, y1, x2, y2 = bbox

    return (
        float(x1),
        float(y1),
        float(x2),
        float(y2)
    )

def shifted_box(box):

    x1, y1, x2, y2 = box

    return (
        x1 + SHIFT_X,
        y1 + SHIFT_Y,
        x2 + SHIFT_X,
        y2 + SHIFT_Y
    )


def intersection(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        x1,
        y1,
        x2,
        y2
    )


def area(box):

    if box is None:
        return 0.0

    x1, y1, x2, y2 = box

    return (
        max(0, x2 - x1)
        *
        max(0, y2 - y1)
    )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 16)
)


for row, case in enumerate(CASES):

    source_path = case["source"]

    if not source_path.exists():

        raise FileNotFoundError(
            source_path
        )

    image = Image.open(source_path).convert("RGB")

    parent_ann = get_annotation(
        case["parent_id"]
    )

    overlap_ann = get_annotation(
        case["overlap_id"]
    )

    parent_box = xywh_to_xyxy(
        parent_ann["bbox"]
    )

    overlap_box = xywh_to_xyxy(
        overlap_ann["bbox"]
    )

    # --------------------------------------------------------
    # Synthetic background crop
    #
    # The generator shifts the parent annotation
    # 100 pixels to the right.
    # --------------------------------------------------------

    synthetic_box = shifted_box(
        parent_box
    )

    # --------------------------------------------------------
    # Calculate actual overlap
    # --------------------------------------------------------

    overlap_region = intersection(
        synthetic_box,
        overlap_box
    )

    overlap_area = area(
        overlap_region
    )

    synthetic_area = area(
        synthetic_box
    )

    overlap_percentage = (
        100.0 * overlap_area / synthetic_area
        if synthetic_area > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Print verification
    # --------------------------------------------------------

    print("\n========================================")
    print(f"CASE {case['case']}")
    print("========================================")

    print("Source:", source_path.name)

    print(
        "Parent annotation:",
        case["parent_id"],
        parent_box
    )

    print(
        "Synthetic background crop:",
        synthetic_box
    )

    print(
        "Overlapping annotation:",
        case["overlap_id"],
        overlap_box
    )

    print(
        f"Overlap area: {overlap_area:.2f} pixels²"
    )

    print(
        f"Crop area: {synthetic_area:.2f} pixels²"
    )

    print(
        f"Overlap percentage: "
        f"{overlap_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Determine visualization region
    # --------------------------------------------------------

    all_boxes = [
        parent_box,
        synthetic_box,
        overlap_box
    ]

    min_x = min(
        b[0] for b in all_boxes
    )

    min_y = min(
        b[1] for b in all_boxes
    )

    max_x = max(
        b[2] for b in all_boxes
    )

    max_y = max(
        b[3] for b in all_boxes
    )

    margin = 120

    crop_x1 = max(
        0,
        int(min_x - margin)
    )

    crop_y1 = max(
        0,
        int(min_y - margin)
    )

    crop_x2 = min(
        image.width,
        int(max_x + margin)
    )

    crop_y2 = min(
        image.height,
        int(max_y + margin)
    )

    source_region = image.crop(
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2
        )
    )

    source_array = np.asarray(
        source_region
    )

    # --------------------------------------------------------
    # Convert source coordinates to local coordinates
    # --------------------------------------------------------

    def local_box(box):

        return (
            box[0] - crop_x1,
            box[1] - crop_y1,
            box[2] - crop_x1,
            box[3] - crop_y1
        )

    local_parent = local_box(
        parent_box
    )

    local_synthetic = local_box(
        synthetic_box
    )

    local_overlap = local_box(
        overlap_box
    )

    local_intersection = (
        local_box(overlap_region)
        if overlap_region is not None
        else None
    )

    # ========================================================
    # LEFT PANEL — SOURCE REGION
    # ========================================================

    ax_left = axes[row, 0]

    ax_left.imshow(
        source_array
    )

    # Parent annotation
    x1, y1, x2, y2 = local_parent

    ax_left.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2.5,
            label="Parent annotation"
        )
    )

    # Synthetic background crop
    x1, y1, x2, y2 = local_synthetic

    ax_left.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=3,
            linestyle="--",
            label="Synthetic background crop"
        )
    )

    # Overlapping annotation
    x1, y1, x2, y2 = local_overlap

    ax_left.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2.5,
            linestyle=":",
            label="Overlapping annotation"
        )
    )

    ax_left.set_title(
        f"Case {case['case']}: Source region",
        fontsize=11,
        fontweight="bold"
    )

    ax_left.axis("off")

    ax_left.legend(
        loc="upper right",
        fontsize=7
    )

    # ========================================================
    # RIGHT PANEL — OVERLAP DETAIL
    # ========================================================

    ax_right = axes[row, 1]

    ax_right.imshow(
        source_array
    )

    # Parent
    x1, y1, x2, y2 = local_parent

    ax_right.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2
        )
    )

    # Synthetic crop
    x1, y1, x2, y2 = local_synthetic

    ax_right.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=3,
            linestyle="--"
        )
    )

    # Overlapping annotation
    x1, y1, x2, y2 = local_overlap

    ax_right.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=3,
            linestyle=":"
        )
    )

    # Highlight intersection
    if local_intersection is not None:

        x1, y1, x2, y2 = local_intersection

        ax_right.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=4
            )
        )

    ax_right.set_title(
        f"Overlap detail: {overlap_area:.0f} px² "
        f"({overlap_percentage:.1f}% of crop)",
        fontsize=10,
        fontweight="bold"
    )

    ax_right.axis("off")


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Source-level geometric audit of synthetic background crops",
    fontsize=16,
    fontweight="bold",
    y=0.995
)

fig.text(
    0.5,
    0.005,
    "Solid box: parent annotation. "
    "Dashed box: proposed synthetic background crop "
    "(100-pixel horizontal shift). "
    "Dotted box: second annotated region.",
    ha="center",
    fontsize=9
)

plt.tight_layout(
    rect=[0, 0.025, 1, 0.98]
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("\n============================================")
print("CORRECTED FIGURE 2 CREATED")
print("============================================")
print(OUT_FILE)