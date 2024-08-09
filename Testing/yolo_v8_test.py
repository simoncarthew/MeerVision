import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the specific YOLOv8 model variant you want to use

# Open the video file
video_path = '/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/camera_trap_footage/fancy_meerkat_test.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (YOLOv8 expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    # Iterate over the detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        # Draw bounding boxes without labels
        for box, score in zip(boxes, scores):
            if score > 0.5:  # Adjust this confidence threshold as needed
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
