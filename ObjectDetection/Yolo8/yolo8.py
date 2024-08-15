import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('ObjectDetection/Yolo8/yolov8n.pt')

# Load the video file
video_path = 'Data/MeerDown/Annotated_videos/22-11-07_C3_06.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was loaded correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict bounding boxes on the current frame
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class label
            label = model.names[cls]

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
