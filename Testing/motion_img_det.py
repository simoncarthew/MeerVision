import cv2
import os

# Create the background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Function to detect and save moving parts
def detect_and_save_moving_parts(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply the background subtractor
        fg_mask = back_sub.apply(frame)
        
        # Find contours of the moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Ignore small contours that might be noise
            if cv2.contourArea(contour) < 500:
                continue
            
            # Get bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            cropped_frame = frame[y:y+h, x:x+w]
            
            # Save the cropped frame
            output_path = os.path.join(output_dir, f"frame_{frame_number}_crop_{i}.jpg")
            cv2.imwrite(output_path, cropped_frame)
        
        frame_number += 1
    
    cap.release()

# Path to the video file and output directory
video_path = '/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/camera_trap_footage/chutney_meerkat_cut.mp4'
output_dir = '/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Testing/Vision/results/mot_det_img'

# Detect and save moving parts
detect_and_save_moving_parts(video_path, output_dir)
