import json
import cv2
import os
from collections import defaultdict

ANNOTATION_FILE = "Data/Observed/annotations.json"
IMAGE_FOLDER = "Data/Observed/frames"

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

def display_camera_trap():
    img_index = 0
    prev_index = 0
    while img_index < len(coco["images"]):
        # get img
        img = coco["images"][img_index]

        # skip image if its not a camera trap image
        ct = True
        if img["camera_trap"] != True:
            prev_index = img_index
            img_index += 1
            ct = False
        
        if ct:
            # get image file name
            image_file = os.path.join(IMAGE_FOLDER,img["file_name"])
            print(image_file)
            image = cv2.imread(image_file)

            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("image", image)

            key = cv2.waitKey(0) & 0xFF

            # Right arrow to go to the next image
            if key == ord('m'):  # Right arrow key
                img_index += 1
            
            # Left arrow to go to the previous image
            elif key == ord('n'):  # Left arrow key
                img_index = prev_index

            # 'q' to quit reviewing
            elif key == ord('q'):
                break

def review_camera_trap_labels():
    # Print the total number of images with camera_trap set to "null"
    no_cam_trap = sum(1 for image in coco["images"] if image["camera_trap"] == "null")
    print("Total missing Camera Trap: " + str(no_cam_trap))

    for image in coco["images"]:
        # Skip images that already have camera_trap set to True or False
        if image["camera_trap"] != "null":
            continue

        # Display the image
        image_file = os.path.join(IMAGE_FOLDER, image["file_name"])
        img = cv2.imread(image_file)

        if img is None:
            print(f"Could not load image: {image_file}")
            continue

        # Draw annotations (bounding boxes) on the image
        image_id = image["id"]
        for annotation in coco["annotations"]:
            if annotation["image_id"] == image_id:
                x, y, width, height = annotation["bbox"]
                # Convert bbox to integer values
                x, y, width, height = int(x), int(y), int(width), int(height)
                # Draw the bounding box on the image
                cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Create a fullscreen window
        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("image", img)

        # Wait for user input
        key = cv2.waitKey(0) & 0xFF

        if key == ord('y'):  # Label as camera trap
            image["camera_trap"] = True
        elif key == ord('n'):  # Label as not a camera trap
            image["camera_trap"] = False
        elif key == ord('q'):  # Quit reviewing
            print("Exiting review...")
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Save the updated annotations
    with open(ANNOTATION_FILE, "w") as f:
        json.dump(coco, f, indent=4)
    print("Annotations updated.")

    # Print the total number of images with camera_trap still set to "null"
    no_cam_trap = sum(1 for image in coco["images"] if image["camera_trap"] == "null")
    print("Total missing Camera Trap: " + str(no_cam_trap))


def save():
    # Save the updated annotations
    with open(ANNOTATION_FILE, "w") as f:
        json.dump(coco, f, indent=4)

def count_annotations_per_class(coco_file_path):
    """Count the number of annotations for each class in the COCO dataset."""
    # Load the COCO JSON file
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary to store counts for each category
    class_counts = defaultdict(int)

    # Iterate over annotations and count occurrences of each category_id
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        class_counts[category_id] += 1

    # Print the counts for each class
    for category in coco_data['categories']:
        class_id = category['id']
        class_name = category['name']
        count = class_counts.get(class_id, 0)
        print(f"{class_name}: {count} annotations")

# do the thing
# count_camera_trap()
count_annotations_per_class("Data/Observed/behaviour_annotations.json")
# display_camera_trap()
# review_camera_trap_labels()