import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.segmentation import watershed
from scipy import ndimage as ndi

def segment_subject(image, bounding_boxes):
    segmented_image = image.copy()
    
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Otsu's method for thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found, keep only the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros(binary.shape, np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Apply the mask to the original ROI
            result = cv2.bitwise_and(roi, roi, mask=mask)
            
            # Draw the contour on the segmented image
            cv2.drawContours(segmented_image[y1:y2, x1:x2], [largest_contour], -1, (0, 255, 0), 2)
    
    return segmented_image

def show_image(image, title="Segmented Image"):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, image_rgb)
    cv2.waitKey(0)  # Wait for a key press to move to the next image
    cv2.destroyAllWindows()

def process_images(image_dir, annotation_csv):
    # Load the annotations
    annotations = pd.read_csv(annotation_csv)
    
    # Iterate over each unique frame in the annotations
    for (video, frame_number), group in annotations.groupby(['video', 'frame_number']):
        image_name = f"{video}_frame_{frame_number}.jpg"
        image_path = os.path.join(image_dir, image_name)
        
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            
            # Extract bounding boxes for this image
            bounding_boxes = group[['x1', 'y1', 'x2', 'y2']].values
            
            # Segment the subjects in the image
            segmented_image = segment_subject(image, bounding_boxes)
            
            # Show the segmented image
            show_image(segmented_image, title=f"Segmented: {image_name}")

# Example usage
image_directory = 'Data/MeerDown/raw/frames'  # Replace with the path to your image directory
annotations_file = 'Data/MeerDown/raw/annotations.csv'  # Replace with the path to your annotations CSV file

process_images(image_directory, annotations_file)
