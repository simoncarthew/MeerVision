import os
import cv2

# Directory containing the videos
DIR = os.path.join("Data", "YoutubeCameraTrap")

# List of video files (without extension)
VIDEOS = ["At the meerkat burrow", "chutney_meerkat_cut", "MeerkatWide0", "MeerkatWide1", "rand_meerkats"]

# Specify the sampling rate (frames per second)
SAMPLING_RATE = 1  # 2 frames per second

# Directory to save the sampled images
SAVE_DIR = "Data/Observed/temp_test_frames"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the save directory if it doesn't exist

# Iterate over all video files
for video in VIDEOS:
    video_path = os.path.join(DIR, video + ".mp4")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval between frames to sample
    frame_interval = int(fps // SAMPLING_RATE)
    
    frame_no = 0  # Frame counter
    saved_frame_no = 0  # Counter for sampled images

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # Break if video is finished
        
        # Save the frame if it matches the interval
        if frame_no % frame_interval == 0:
            # Construct the output image filename
            image_filename = f"{video}_{saved_frame_no}.jpg"
            image_path = os.path.join(SAVE_DIR, image_filename)
            
            # Save the frame as an image
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")
            saved_frame_no += 1
        
        frame_no += 1

    cap.release()

print("Sampling complete.")