import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the model to evaluation mode
model.eval()

# Open the video file
video_path = '/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Testing/Vision/meerkat_test.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (YOLOv5 expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    # Get the bounding boxes
    boxes = results.xyxy[0].cpu().numpy()

    # Draw bounding boxes without labels
    for box in boxes:
        x1, y1, x2, y2, conf, _ = box
        if conf > 0.5:  # You can adjust this confidence threshold
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()