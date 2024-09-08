import json

ANNOTATION_FILE = "Data/Observed/annotations.json"

# Load existing annotations
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

def add_camera_trap():
    # Add 'camera_trap' field to existing images
    for image in coco["images"]:
        image["camera_trap"] = "null"  # or any other default value

def count_camera_trap():
    no_cam_trap = 0
    for image in coco["images"]:
        if (image["camera_trap"] == True): no_cam_trap += 1 
    print("Total camera trap (test) images: " + str(no_cam_trap))


def save():
    # Save the updated annotations
    with open(ANNOTATION_FILE, "w") as f:
        json.dump(coco, f, indent=4)

# do the thing
count_camera_trap()