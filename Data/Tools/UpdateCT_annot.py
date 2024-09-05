import json

ANNOTATION_FILE = "Data/Observed/annotations.json"

# Load existing annotations
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

# Add 'camera_trap' field to existing images
for image in coco["images"]:
    image["camera_trap"] = "null"  # or any other default value

# Save the updated annotations
with open(ANNOTATION_FILE, "w") as f:
    json.dump(coco, f, indent=4)