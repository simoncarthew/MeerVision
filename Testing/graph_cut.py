import os
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

def apply_graph_cut(image):
    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Reshape the image
    pixel_values = image.reshape((-1, 3))
    
    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(pixel_values)
    
    # Predict labels
    labels = gmm.predict(pixel_values)
    
    # Reshape labels back to image shape
    segmented = labels.reshape(image.shape[:2])
    
    # Create a binary mask
    mask = np.zeros(segmented.shape, dtype=np.uint8)
    mask[segmented == 1] = 255
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def trace_foreground(image, mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result

def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            # Apply graph cut
            mask = apply_graph_cut(image)
            
            # Trace foreground
            result = trace_foreground(image, mask)
            
            # Save the result
            output_path = os.path.join(output_folder, f"traced_{filename}")
            cv2.imwrite(output_path, result)
            
            print(f"Processed: {filename}")

# Usage
input_folder = "/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/test/meerkat"
output_folder = "/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Testing/Vision/results"
process_images(input_folder, output_folder)