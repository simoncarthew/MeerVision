import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image

# Initialize SAM model
sam_checkpoint = "Data/Segments/sam_vit_b_01ec64.pth"
model_type = "vit_b"  # Change to your desired model type, e.g. vit_h, vit_l
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_predictor = SamPredictor(sam)

# Global variables to store bounding box
bbox = []
drawing = False

# Function to save the segmented object as PNG with transparency
def save_segmented_image(image, mask, output_path):
    mask = mask.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)

    # Crop image and mask to the bounding box
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # Create transparent background
    alpha = np.zeros(cropped_mask.shape, dtype=np.uint8)
    alpha[cropped_mask > 0] = 255

    # Merge the RGB image with the alpha channel
    segmented_image = np.dstack((cropped_image, alpha))

    # Save as PNG with transparency
    im_pil = Image.fromarray(segmented_image)
    im_pil.save(output_path, "PNG")

# Mouse callback function to draw the bounding box
def draw_bbox(event, x, y, flags, param):
    global bbox, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        bbox = [(x, y)]  # Start point

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = param.copy()  # Create a copy to update the box as you drag
            cv2.rectangle(img_copy, bbox[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox.append((x, y))  # End point
        cv2.rectangle(param, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow('Image', param)

# Function to let user draw a bounding box and segment the region
def interactive_segmentation(image_path, output_dir):
    global bbox
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Set the image for SAM predictor
    sam_predictor.set_image(original_image)

    # Display image and set mouse callback to draw the bounding box
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', draw_bbox, image)

    # Wait for user to press Enter after drawing the bounding box
    print("Draw a bounding box around the object and press 'Enter' when done.")
    while True:
        key = cv2.waitKey(1)
        if key == 13:  # Enter key to confirm the bounding box
            break

    # Extract coordinates from the bounding box
    if len(bbox) == 2:
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]

        # Convert bbox to format (x, y, w, h) for SAM input
        input_box = np.array([min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)])

        # Predict mask for the region inside the bounding box
        masks, _, _ = sam_predictor.predict(
            box=input_box, multimask_output=False
        )

        mask = masks[0]
        output_file = f"{output_dir}/segmented_{x1}_{y1}.png"
        save_segmented_image(original_image, mask, output_file)
        print(f"Segmented object saved to {output_file}")

    cv2.destroyAllWindows()

# Main function to process a folder of images
def process_images(image_folder, output_folder):
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing {image_name}...")
            interactive_segmentation(image_path, output_folder)

# Run the script
image_folder = "Data/Segments"
output_folder = "Data/Segments/Segmented"
process_images(image_folder, output_folder)
