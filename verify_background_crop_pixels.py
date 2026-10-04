
from pathlib import Path
import json
import math
import pandas as pd
from PIL import Image, ImageDraw, ImageChops

ROOT = Path(".")
REPORT = ROOT / "results_midog_source_analysis/background_crop_overlap_audit.csv"
JSON_PATH = ROOT / "data/raw/MIDOG/MIDOGpp.json"
SOURCE_DIR = ROOT / "data/raw/MIDOG/domain1a"
OUT_DIR = ROOT / "results_midog_source_analysis/background_crop_visual_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

images_by_name = {im["file_name"]: im for im in metadata["images"]}
annotations_by_image = {}
for ann in metadata["annotations"]:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)

categories = {c["id"]: c["name"] for c in metadata["categories"]}
report = pd.read_csv(REPORT)
flagged = report[report["status"] == "POTENTIAL_OVERLAP"]

summary = []

for _, row in flagged.iterrows():
    source_name = str(row["source_file"])
    source_path = SOURCE_DIR / source_name
    crop_path = ROOT / Path(str(row["crop_path"]))

    print("\n" + "=" * 60)
    print("Crop:", crop_path)
    print("Source:", source_path)

    if not source_path.exists() or not crop_path.exists():
        print("MISSING SOURCE OR SAVED CROP; skipping.")
        continue

    source = Image.open(source_path).convert("RGB")
    saved_crop = Image.open(crop_path).convert("RGB")

    x1 = int(round(float(row["candidate_x1"])))
    y1 = int(round(float(row["candidate_y1"])))
    x2 = int(round(float(row["candidate_x2"])))
    y2 = int(round(float(row["candidate_y2"])))

    # Compare saved crop with the corresponding original-image region.
    reconstructed = source.crop((x1, y1, x2, y2))
    same_size = reconstructed.size == saved_crop.size
    pixels_match = (
        same_size
        and ImageChops.difference(
            reconstructed, saved_crop
        ).getbbox() is None
    )

    print("Saved crop dimensions:", saved_crop.size)
    print("Reconstructed dimensions:", reconstructed.size)
    print("Saved pixels match source region:", pixels_match)

    image_record = images_by_name.get(source_name)
    if image_record is None:
        print("Source metadata record not found.")
        continue

    anns = annotations_by_image.get(image_record["id"], [])
    matches = []

    for ann in anns:
        ax1, ay1, ax2, ay2 = ann["bbox"]
        ix1, iy1 = max(x1, ax1), max(y1, ay1)
        ix2, iy2 = min(x2, ax2), min(y2, ay2)
        area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        if area > 0:
            matches.append(ann)
            print(
                "Overlapping annotation:",
                ann["id"],
                "|",
                categories.get(ann["category_id"], "unknown"),
                "| bbox:",
                ann["bbox"],
                "| intersection:",
                area
            )

    # Display a 200 x 200 source region around the candidate crop.
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx1, ry1 = max(0, cx - 100), max(0, cy - 100)
    rx2, ry2 = min(source.width, cx + 100), min(source.height, cy + 100)

    context = source.crop((rx1, ry1, rx2, ry2))
    context.thumbnail((400, 400))

    # Draw boxes on a separate, resized context image.
    scale_x = context.width / (rx2 - rx1)
    scale_y = context.height / (ry2 - ry1)
    context_panel = context.copy()
    draw = ImageDraw.Draw(context_panel)

    def local_box(box):
        bx1, by1, bx2, by2 = box
        return (
            (bx1 - rx1) * scale_x,
            (by1 - ry1) * scale_y,
            (bx2 - rx1) * scale_x,
            (by2 - ry1) * scale_y,
        )

    draw.rectangle(local_box((x1, y1, x2, y2)),
                   outline="lime", width=3)

    for ann in matches:
        draw.rectangle(local_box(ann["bbox"]),
                       outline="red", width=3)

    # Enlarge the candidate and saved crop for visual inspection.
    candidate_view = reconstructed.resize((200, 200))
    saved_view = saved_crop.resize((200, 200))

    panel_width = 420
    panel_height = max(440, context_panel.height + 100)
    panel = Image.new("RGB", (panel_width * 3, panel_height), "white")
    d = ImageDraw.Draw(panel)

    d.text((10, 10), f"{source_name}: surrounding source region", fill="black")
    d.text((panel_width + 10, 10), "Candidate from source", fill="black")
    d.text((2 * panel_width + 10, 10), "Saved background crop", fill="black")

    panel.paste(context_panel, (10, 35))
    panel.paste(candidate_view, (panel_width + 10, 35))
    panel.paste(saved_view, (2 * panel_width + 10, 35))

    draw_panel = ImageDraw.Draw(panel)
    draw_panel.text((10, panel_height - 45),
                    "Green = candidate crop; red = overlapping annotation",
                    fill="black")
    draw_panel.text((panel_width + 10, panel_height - 45),
                    f"Pixel match: {pixels_match}", fill="black")

    output = OUT_DIR / f"{crop_path.stem}_verification.png"
    panel.save(output)
    print("Visual report:", output)

    summary.append({
        "crop": str(crop_path),
        "source": str(source_path),
        "pixels_match": pixels_match,
        "overlap_count": len(matches),
        "visual_report": str(output),
    })

summary_path = OUT_DIR / "verification_summary.csv"
pd.DataFrame(summary).to_csv(summary_path, index=False)
print("\nVerification summary:", summary_path)
print("Completed:", len(summary), "crops")