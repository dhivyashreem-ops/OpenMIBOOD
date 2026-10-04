
import json
import csv
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "data" / "raw" / "MIDOG" / "MIDOGpp.json"
CROP_ROOT = ROOT / "data" / "midog" / "1a"
OUT_PATH = ROOT / "results_midog_source_analysis" / "background_crop_overlap_audit.csv"

# Read metadata without changing the dataset.
with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

images_by_id = {img["id"]: img for img in data["images"]}
annotations_by_image = {}

for ann in data["annotations"]:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)


def intersects(a, b):
    """Strict positive-area intersection of [x1,y1,x2,y2] rectangles."""
    return (
        a[0] < b[2] and a[2] > b[0]
        and a[1] < b[3] and a[3] > b[1]
    )


def annotation_bbox(ann):
    """MIDOG bbox is [x1, y1, x2, y2]."""
    return list(ann["bbox"])


rows = []
missing_sources = []
checked = 0
not_found = 0

for crop_path in CROP_ROOT.rglob("*_0.tiff"):
    # Expected filename: sourceID_annotationID_0.tiff
    parts = crop_path.stem.split("_")
    if len(parts) != 3:
        rows.append({
            "crop_path": str(crop_path.relative_to(ROOT)),
            "status": "UNPARSEABLE_FILENAME",
        })
        continue

    source_id_text, annotation_id_text, _ = parts

    try:
        source_id = int(source_id_text)
        annotation_id = int(annotation_id_text)
    except ValueError:
        rows.append({
            "crop_path": str(crop_path.relative_to(ROOT)),
            "status": "UNPARSEABLE_FILENAME",
        })
        continue

    source_filename = f"{source_id:03d}.tiff"
    matching_images = [
        img for img in data["images"]
        if Path(img.get("file_name", "")).name == source_filename
    ]

    if len(matching_images) != 1:
        missing_sources.append(source_filename)
        rows.append({
            "crop_path": str(crop_path.relative_to(ROOT)),
            "source_file": source_filename,
            "status": "SOURCE_METADATA_NOT_UNIQUELY_FOUND",
        })
        continue

    source_image = matching_images[0]
    source_annotations = annotations_by_image.get(source_image["id"], [])

    # Find the annotation used to generate this background crop.
    parent_annotations = [
        ann for ann in source_annotations
        if ann["id"] == annotation_id
    ]

    if len(parent_annotations) != 1:
        rows.append({
            "crop_path": str(crop_path.relative_to(ROOT)),
            "source_file": source_filename,
            "parent_annotation_id": annotation_id,
            "status": "PARENT_ANNOTATION_NOT_UNIQUELY_FOUND",
        })
        continue

    parent_bbox = annotation_bbox(parent_annotations[0])
    candidate = [
        parent_bbox[0] + 100,
        parent_bbox[1],
        parent_bbox[2] + 100,
        parent_bbox[3],
    ]

    # Check the saved crop's dimensions.
    try:
        with Image.open(crop_path) as im:
            crop_size = im.size
    except Exception as exc:
        rows.append({
            "crop_path": str(crop_path.relative_to(ROOT)),
            "source_file": source_filename,
            "status": f"CROP_READ_ERROR: {type(exc).__name__}",
        })
        continue

    # Compare against every annotation in the source image.
    overlapping_ids = [
        ann["id"] for ann in source_annotations
        if intersects(candidate, annotation_bbox(ann))
    ]

    checked += 1
    rows.append({
        "crop_path": str(crop_path.relative_to(ROOT)),
        "source_file": source_filename,
        "parent_annotation_id": annotation_id,
        "candidate_x1": candidate[0],
        "candidate_y1": candidate[1],
        "candidate_x2": candidate[2],
        "candidate_y2": candidate[3],
        "crop_width": crop_size[0],
        "crop_height": crop_size[1],
        "overlapping_annotation_ids": ";".join(map(str, overlapping_ids)),
        "overlap_count": len(overlapping_ids),
        "status": (
            "POTENTIAL_OVERLAP"
            if overlapping_ids
            else "NO_ANNOTATED_OVERLAP"
        ),
    })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fieldnames = sorted({key for row in rows for key in row})
with OUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

potential = sum(
    row.get("status") == "POTENTIAL_OVERLAP" for row in rows
)

print("MIDOG background-crop audit complete")
print("Background crop files found:", len(rows))
print("Crops checked against source annotations:", checked)
print("Potential overlaps:", potential)
print("Source metadata lookup issues:", len(missing_sources))
print("Report:", OUT_PATH)
print("Original dataset and labels were not modified.")