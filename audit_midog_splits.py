
from pathlib import Path
from collections import Counter, defaultdict
import re

SPLIT_FILES = {
    "train": Path("data/benchmark_imglist/midog/train_midog.txt"),
    "val": Path("data/benchmark_imglist/midog/valid_midog.txt"),
    "test": Path("data/benchmark_imglist/midog/test_midog.txt"),
}

def read_split(path):
    rows = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue

        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed line {line_no} in {path}: {line}")

        image_path, label_text = parts
        image_path = image_path.replace("\\", "/")
        match = re.search(r"/([^/]+)/([^/]+)\.tiff$", "/" + image_path)

        # Candidate grouping IDs only; confirm their meaning from
        # the MIDOG dataset documentation before interpreting them.
        parent_id = Path(image_path).parent.name
        filename = Path(image_path).stem
        filename_prefix = filename.split("_")[0]

        rows.append({
            "path": image_path,
            "label": int(label_text),
            "parent_id": parent_id,
            "filename_prefix": filename_prefix,
        })
    return rows

splits = {}
for name, path in SPLIT_FILES.items():
    if not path.exists():
        print(f"MISSING: {path}")
        continue
    splits[name] = read_split(path)

print("\n=== SPLIT COUNTS ===")
for name, rows in splits.items():
    print(f"{name}: {len(rows)} crops")
    print("  classes:", dict(sorted(Counter(r["label"] for r in rows).items())))
    print("  candidate parent IDs:", len(set(r["parent_id"] for r in rows)))

print("\n=== CANDIDATE PARENT-ID OVERLAP ===")
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    if a not in splits or b not in splits:
        continue
    ids_a = {r["parent_id"] for r in splits[a]}
    ids_b = {r["parent_id"] for r in splits[b]}
    overlap = sorted(ids_a & ids_b)
    print(f"{a} vs {b}: {len(overlap)} shared parent IDs")
    if overlap:
        print("  examples:", overlap[:20])

print("\n=== DUPLICATE PATH CHECK ===")
for name, rows in splits.items():
    paths = [r["path"] for r in rows]
    duplicates = [p for p, n in Counter(paths).items() if n > 1]
    print(f"{name}: {len(duplicates)} duplicated paths")

if all(k in splits for k in ("train", "val", "test")):
    sets = {
        name: {r["path"] for r in rows}
        for name, rows in splits.items()
    }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"{a}/{b} exact path overlap:", len(sets[a] & sets[b]))

print("\nAudit complete. Candidate IDs require documentation verification.")