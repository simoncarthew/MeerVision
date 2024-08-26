import os
import cv2
import torch
from PytorchWildlife.models import detection as pw_detection

# Initialize the MegaDetectorV5 model
detection_model = pw_detection.MegaDetectorV5()  # Model weights are automatically downloaded.

# Directory containing the images
image_dir = "Data/MeerDown/raw/frames"

# Define the standard size for all images (e.g., 1280x1280)
standard_size = (1280, 1280)

# Loop through each image in the directory
for image_file in os.listdir(image_dir):
    # Create the full path to the image file
    image_path = os.path.join(image_dir, image_file)
    
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Resize the image to the standard size
    img_resized = cv2.resize(img, standard_size)

    # Convert the image to a PyTorch tensor and permute it to match the model's input format
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()

    # Run the detection model
    detection_result = detection_model.single_image_detection(img_tensor)

    # Print the detection result (bounding boxes, scores, etc.)
    print(detection_result)

    # Display the image with detections (you can overlay bounding boxes using OpenCV)
    cv2.imshow('Animal Detection', img_resized)

    # Wait for a key press to move to the next image
    cv2.waitKey(0)

# Close all OpenCV windows after processing
cv2.destroyAllWindows()