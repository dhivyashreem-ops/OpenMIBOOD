import json

JSON_PATH = "data/raw/MIDOG/MIDOGpp.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# First 50 images = Domain 1a according to OpenMIBOOD
images = data["images"][:50]
image_map = {img["id"]: img for img in images}

annotations_by_image = {img_id: [] for img_id in image_map}

for ann in data["annotations"]:
    if ann["image_id"] in annotations_by_image:
        annotations_by_image[ann["image_id"]].append(ann)


def validate_patch(img_dims, annotations, additional_bbox):
    # Same logic as official OpenMIBOOD code
    if (
        additional_bbox[0] < 0
        or additional_bbox[1] < 0
        or additional_bbox[2] >= img_dims[0]
        or additional_bbox[3] >= img_dims[1]
    ):
        return False

    for annotation in annotations:
        bbox = annotation["bbox"]

        if (
            bbox[0] < additional_bbox[0] < bbox[2]
            and bbox[1] < additional_bbox[1] < bbox[3]
        ):
            return False

        if (
            bbox[0] < additional_bbox[2] < bbox[2]
            and bbox[1] < additional_bbox[3] < bbox[3]
        ):
            return False

    return True


original_patches = 0
mitotic = 0
not_mitotic = 0
additional_patches = 0

for image_id, annotations in annotations_by_image.items():

    img = image_map[image_id]

    width = img["width"]
    height = img["height"]

    for annotation in annotations:

        bbox = annotation["bbox"]

        # Official OpenMIBOOD condition
        if (
            bbox[2] - bbox[0] == 50
            and bbox[3] - bbox[1] == 50
        ):

            original_patches += 1

            if annotation["category_id"] == 1:
                mitotic += 1

            elif annotation["category_id"] == 2:
                not_mitotic += 1

            # Official additional/background patch
            additional_bbox = [
                bbox[0] + 100,
                bbox[1],
                bbox[2] + 100,
                bbox[3],
            ]

            if validate_patch(
                (width, height),
                annotations,
                additional_bbox,
            ):
                additional_patches += 1


print()
print("=" * 55)
print("OPENMIBOOD DOMAIN 1a DATASET AUDIT")
print("=" * 55)

print(f"Source images:                    {len(images)}")
print(f"Annotations:                     {sum(len(v) for v in annotations_by_image.values())}")
print(f"Original 50x50 patches:           {original_patches}")
print(f"  Mitotic patches:                {mitotic}")
print(f"  Not-mitotic patches:            {not_mitotic}")
print(f"Valid additional label-0 patches: {additional_patches}")
print(f"Total potential patches:          {original_patches + additional_patches}")

print("=" * 55)