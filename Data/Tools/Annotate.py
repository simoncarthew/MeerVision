import os
import json
import cv2
import glob

# Folder paths
IMAGE_FOLDER = "Data/Observed/frames"
ANNOTATION_FILE = "Data/Observed/annotations.json"

# see previous annotations 
REVIEW_ANNOTATED_IMAGES = True

# COCO structure
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "meerkat"
        }
    ]
}

# Load existing annotations
if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, "r") as f:
        coco = json.load(f)

image_id = len(coco["images"]) + 1
annotation_id = len(coco["annotations"]) + 1
drawing = False
start_x, start_y = -1, -1
boxes = []

# calculate stats
images_completed = len(coco["images"])
image_files = os.listdir(IMAGE_FOLDER)
total_images = sum(os.path.isfile(os.path.join(IMAGE_FOLDER, item)) for item in image_files)
remaining_images = total_images - images_completed
percentage_complete = images_completed / total_images * 100

# display stats
print("Total annotated images:", images_completed)
print("Remaining images:", remaining_images)
print("Percentage complete: " + str(percentage_complete) + "%")
print()

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = image.copy()
            cv2.rectangle(temp_image, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        boxes.append((start_x, start_y, x, y))
        cv2.rectangle(image, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv2.imshow("image", image)

def save_annotation(image_name, boxes, camera_trap=False):
    global image_id, annotation_id

    # Save image 'info'
    image_info = {
        "id": image_id,
        "file_name": image_name,
        "height": image.shape[0],
        "width": image.shape[1],
        "camera_trap": camera_trap
    }
    coco["images"].append(image_info)

    # Save each bounding box
    for box in boxes:
        x_min = min(box[0], box[2])
        y_min = min(box[1], box[3])
        width = abs(box[2] - box[0])
        height = abs(box[3] - box[1])

        # Ensure the bounding box stays within the image boundaries
        x_min = max(0, min(x_min, image.shape[1]))
        y_min = max(0, min(y_min, image.shape[0]))
        width = min(width, image.shape[1] - x_min)
        height = min(height, image.shape[0] - y_min)

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # meerkat class
            "bbox": [x_min, y_min, width, height],
            "area": width * height,
            "iscrowd": 0
        }
        coco["annotations"].append(annotation)
        annotation_id += 1

    # Update image_id
    image_id += 1

# Load existing annotations
if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, "r") as f:
        coco = json.load(f)

def review_annotated_images():
    # Get annotated image filenames from the COCO data
    annotated_image_files = {img["file_name"] for img in coco["images"]}
    annotated_image_files = [os.path.join(IMAGE_FOLDER, fname) for fname in annotated_image_files]
    annotated_image_files.sort()
    current_index = 0

    while True:
        if current_index >= len(annotated_image_files):
            print("No more annotated images.")
            break
        
        image_file = annotated_image_files[current_index]
        image_name = os.path.basename(image_file)
        image = cv2.imread(image_file)

        # Draw bounding boxes on the image
        image_id = next(img["id"] for img in coco["images"] if img["file_name"] == image_name)
        annotations = [ann for ann in coco["annotations"] if ann["image_id"] == image_id]
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("image", image)

        key = cv2.waitKey(0) & 0xFF

        # Right arrow to go to the next image
        if key == ord('m'):  # Right arrow key
            current_index += 1
        
        # Left arrow to go to the previous image
        elif key == ord('n'):  # Left arrow key
            current_index = max(current_index - 1, 0)

        # 'q' to quit reviewing
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

# Main annotation process
if REVIEW_ANNOTATED_IMAGES:
    review_annotated_images()

# Get the list of unannotated images
annotated_images = {img["file_name"] for img in coco["images"]}
image_files = [f for f in glob.glob(os.path.join(IMAGE_FOLDER, "*")) if os.path.basename(f) not in annotated_images]

num_new_images = 0
new_image_annotations = []
for image_file in image_files:
    image_name = os.path.basename(image_file)
    image = cv2.imread(image_file)
    boxes = []

    # Create a named window
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", draw_rectangle)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Press 's' to save annotations and move to the next image
        if key == ord("s") or key == ord("a"):
            camera_trap = (key == ord('a'))
            save_annotation(image_name, boxes, camera_trap)
            num_new_images += 1
            new_image_annotations.append(image_file)
            break

        # Press 'd' to delete the image if it doesn't contain a meerkat
        elif key == ord("d"):
            print("Are you sure you want to delete this image? Press 'y' to confirm.")
            if cv2.waitKey(0) & 0xFF == ord('y'):
                os.remove(image_file)
                print(f"{image_name} deleted.")
                break
            else:
                print("Deletion canceled.")
                continue

        # Press 'r' to reset the bounding boxes
        elif key == ord("r"):
            boxes = []
            image = cv2.imread(image_file)
            cv2.imshow("image", image)

        # Press 'u' to undo the last bounding box
        elif key == ord("u") and boxes:
            boxes.pop()  # Remove the last bounding box
            image = cv2.imread(image_file)  # Reload the image
            # Redraw remaining boxes
            for box in boxes:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow("image", image)

        # Press 'q' to quit the program
        elif key == ord("q"):
            break

    # Break loop if 'q' was pressed
    if key == ord("q"):
        break

# review after complettion
current_index = 0

while True:
    if current_index >= len(new_image_annotations):
        print("No more annotated images.")
        break
    
    image_file = new_image_annotations[current_index]
    image_name = os.path.basename(image_file)
    image = cv2.imread(image_file)

    # Draw bounding boxes on the image
    image_id = next(img["id"] for img in coco["images"] if img["file_name"] == image_name)
    annotations = [ann for ann in coco["annotations"] if ann["image_id"] == image_id]
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("image", image)

    key = cv2.waitKey(0) & 0xFF

    # Right arrow to go to the next image
    if key == ord('m'):  # Right arrow key
        current_index += 1
    
    # Left arrow to go to the previous image
    elif key == ord('n'):  # Left arrow key
        current_index = max(current_index - 1, 0)

    # 'q' to quit reviewing
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

# calculate stats
images_completed =  images_completed + num_new_images
remaining_images = remaining_images - num_new_images
percentage_complete = images_completed / total_images * 100

# display stats
print()
print("New annotations:", num_new_images)
print("Total annotated images:", images_completed)
print("Remaining images:", remaining_images)
print("Percentage complete: " + str(percentage_complete) + "%")

# Save COCO annotations to file
with open(ANNOTATION_FILE, "w") as f:
    json.dump(coco, f, indent=4)

cv2.destroyAllWindows()
