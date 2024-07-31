import cv2
import numpy as np
from sort import Sort

# Initialize video capture
cap = cv2.VideoCapture('/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/camera_trap_footage/rand_meerkats.mp4')

# Initialize background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2()

# Initialize SORT tracker
tracker = Sort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = back_sub.apply(frame)

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare detections for SORT (x1, y1, x2, y2, score)
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detections.append([x, y, x+w, y+h, 1.0])

    # Update tracker
    trackers = tracker.update(np.array(detections))

    # Draw bounding boxes
    for d in trackers:
        x1, y1, x2, y2, obj_id = map(int, d[:5])  # Ensure only the first five elements are used
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
