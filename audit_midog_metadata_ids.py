
import json
import re
from pathlib import Path

metadata_path = Path("data/raw/MIDOG/MIDOGpp.json")

split_files = {
    "train": Path("data/benchmark_imglist/midog/train_midog.txt"),
    "val": Path("data/benchmark_imglist/midog/valid_midog.txt"),
    "test": Path("data/benchmark_imglist/midog/test_midog.txt"),
}

with metadata_path.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

# Original-image identifiers listed in the metadata
metadata_ids = {
    Path(image["file_name"]).stem
    for image in metadata["images"]
}

print("Metadata source images:", len(metadata_ids))

split_ids = {}

for split, list_path in split_files.items():
    rows = [
        line.split()[0].replace("\\", "/")
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Expected path structure: domain/source_id/image_name
    found_ids = set()

    for path in rows:
        match = re.match(r"[^/]+/([^/]+)/", path)
        if match:
            found_ids.add(match.group(1))

    missing_ids = sorted(found_ids - metadata_ids)
    split_ids[split] = found_ids

    print(f"\n{split.upper()}")
    print("Patch count:", len(rows))
    print("Unique source IDs:", len(found_ids))
    print("IDs missing from metadata:", len(missing_ids))

    if missing_ids:
        print("Missing examples:", missing_ids[:10])

print("\nCROSS-SPLIT SOURCE-ID OVERLAP")

for first, second in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]:
    overlap = split_ids[first] & split_ids[second]
    print(f"{first} vs {second}: {len(overlap)} shared IDs")

print("\nAudit completed.")
print(
    "Note: Matching metadata IDs verifies source-image identifiers, "
    "not patient-level independence."
)