import json
import os

ANNOTATION_FILE = "Data/Observed/annotations.json"

# Load existing annotations
if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, "r") as f:
        coco = json.load(f)

# Iterate through all annotations to check bounding boxes
for annotation in coco["annotations"]:
    # Get the image corresponding to this annotation
    image_info = next(img for img in coco["images"] if img["id"] == annotation["image_id"])
    image_width = image_info["width"]
    image_height = image_info["height"]

    # Get the bounding box
    x_min, y_min, width, height = annotation["bbox"]

    # Ensure the bounding box stays within the image boundaries
    x_min = max(0, min(x_min, image_width))
    y_min = max(0, min(y_min, image_height))
    width = min(width, image_width - x_min)
    height = min(height, image_height - y_min)

    # Update the annotation with the corrected bounding box
    annotation["bbox"] = [x_min, y_min, width, height]
    annotation["area"] = width * height  # Update the area as well

# Save corrected annotations back to the file
with open(ANNOTATION_FILE, "w") as f:
    json.dump(coco, f, indent=4)

print("Annotations have been corrected and saved.")
