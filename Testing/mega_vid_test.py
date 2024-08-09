import cv2
import torch
from PytorchWildlife.models import detection as pw_detection

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Load MegaDetector v5
detection_model = pw_detection.MegaDetectorV5(device=device)

# Open the video file
video_path = "/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/camera_trap_footage/chutney_meerkat_cut.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")


# Create VideoWriter object to save the output video
out = cv2.VideoWriter('/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Testing/Vision/results/chutney_meerkats_mega.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % 1 == 0:
        resized_frame = cv2.resize(frame, (1280, 1280))
        img_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            detection_result = detection_model.single_image_detection(img_tensor[0])
        
        detections = detection_result['detections']
        print(detections)
        
        for i in range(len(detections.confidence)):
            category = detections.class_id[i]  # Class ID
            bbox = detections.xyxy[i]  # Bounding box
            conf = detections.confidence[i]  # Confidence score
            
            # Scale back to original size
            x1, y1, x2, y2 = [int(coord * width / 1280) if j % 2 == 0 else int(coord * height / 1280) for j, coord in enumerate(bbox)]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Animal: {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Write the frame to the output video
        out.write(frame)
    
    frame_count += 1
    frames_left = total_frames - frame_count
    print(f"Frames left to process: {frames_left}")

# Release video capture and writer objects
cap.release()
out.release()
print("Video processing complete. Output saved as 'rand_meerkats_mega.mp4'")