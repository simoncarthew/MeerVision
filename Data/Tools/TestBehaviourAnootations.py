import json
import os
import cv2

# Load the COCO JSON file
coco_file_path = "Data/Observed/annotations.json"  # Path to your COCO file
image_dir = "Data/Observed/frames"  # Directory containing your images
output_coco_file = "Data/Observed/behaviour_annotations.json"  # Output file path for saving new annotations
from tqdm import tqdm

# Load the binary JSON data
with open(coco_file_path, 'r') as f:
    bin_coco_data = json.load(f)

# Get camera trap images
ct_images = [img for img in bin_coco_data['images'] if img.get('camera_trap', False)]
image_ids = {img['id'] for img in ct_images}
bin_annotations = [ann for ann in bin_coco_data['annotations'] if ann['image_id'] in image_ids]

# Load the behavior annotations JSON data, or create a new structure if it doesn't exist
if os.path.exists(output_coco_file):
    with open(output_coco_file, 'r') as f:
        beh_coco_data = json.load(f)
else:
    beh_coco_data = {
        "images": ct_images,
        "annotations": [],
        "categories": [
            {"id": 0, "name": "vigilant"},
            {"id": 1, "name": "foraging"},
            {"id": 2, "name": "other"}
        ]
    }

# Get the set of processed annotation IDs
processed_annotation_ids = {ann['id'] for ann in beh_coco_data['annotations']}

# Filter for unprocessed binary annotations
unprocessed_annotations = [ann for ann in bin_annotations if ann['id'] not in processed_annotation_ids]
print("Remaining annotations:" + str(len(unprocessed_annotations)))

def display_annotation(image_path, bbox):
    """Display the annotation's cropped image region using OpenCV."""
    img = cv2.imread(image_path)
    x, y, w, h = bbox
    cropped_img = img[int(y):int(y + h), int(x):int(x + w)]
    if cropped_img.size == 0:
        print(f"Error: Cropped image is empty for bbox {bbox} in image {image_path}")
        return None
    cv2.imshow("Annotation", cropped_img)
    return cropped_img

# Iterate through filtered unprocessed annotations
for img_info in ct_images:
    img_id = img_info['id']
    img_file_name = img_info['file_name']
    image_path = os.path.join(image_dir, img_file_name)

    # Get the unprocessed annotations for this image
    annotations = [ann for ann in unprocessed_annotations if ann['image_id'] == img_id]

    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        cropped_img = display_annotation(image_path, bbox)

        # Wait for keypress to assign a new category
        key = cv2.waitKey(0)

        if key == ord('v'):
            ann['category_id'] = 0  # vigilant
        elif key == ord('f'):
            ann['category_id'] = 1  # foraging
        elif key == ord('o'):
            ann['category_id'] = 2  # other
        elif key == ord('q'):
            # Quit and save progress
            with open(output_coco_file, 'w') as outfile:
                json.dump(beh_coco_data, outfile, indent=4)
            print("Progress saved. Exiting.")
            cv2.destroyAllWindows()
            exit()

        # Append the newly labeled annotation to the output data
        beh_coco_data['annotations'].append(ann)

# Save the updated behavior annotations COCO file
with open(output_coco_file, 'w') as outfile:
    json.dump(beh_coco_data, outfile, indent=4)

cv2.destroyAllWindows()
print("Finished labeling all annotations.")
