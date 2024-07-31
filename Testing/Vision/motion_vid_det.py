import cv2
import os

# Create the background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Function to detect and save moving parts with bounding boxes in a new video
def detect_and_save_with_bounding_boxes(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the background subtractor
        fg_mask = back_sub.apply(frame)

        # Find contours of the moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Ignore small contours that might be noise
            if cv2.contourArea(contour) < 500:
                continue

            # Get bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw bounding box on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    cap.release()
    out.release()

# Path to the input video file and output video file
video_path = '/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/camera_trap_footage/chutney_meerkat_cut.mp4'
output_video_path = '/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Testing/Vision/results/chutney_meerkat_mot.mp4'

# Detect and save with bounding boxes
detect_and_save_with_bounding_boxes(video_path, output_video_path)
