import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_edge_detection(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale for edge detection
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Define edge detection kernel (Sobel operator in x and y directions)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel kernel for x direction
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel kernel for y direction
    
    # Apply the Sobel filter
    edges_x = cv2.filter2D(img, cv2.CV_64F, sobel_x)  # Convert to float64
    edges_y = cv2.filter2D(img, cv2.CV_64F, sobel_y)  # Convert to float64
    
    # Compute the magnitude of gradients
    edges = cv2.magnitude(edges_x, edges_y)
    
    # Normalize the result for saving as an image
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    edges = edges.astype(np.uint8)
    
    # Increase the brightness
    edges = cv2.add(edges, np.full(edges.shape, 50, dtype=np.uint8))  # Add brightness

    # Save the edge-detected image as a PNG file
    cv2.imwrite("Data/ReportImages/edges.png", edges)

    print(f"Edge detection image saved")
    return edges

if __name__ == "__main__":
    image_path = 'Data/YoutubeCameraTrap/meerkat.png'  # Replace with the path to your image
    edges = apply_edge_detection(image_path)
