import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture('/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/camera_trap_footage/rand_meerkats.mp4')

# Create background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Dictionary to store trackers and their last known positions
trackers = {}

# Counter for assigning unique IDs to objects
next_object_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction``
    fgMask = backSub.apply(frame)
    
    # Threshold the mask
    thresh = cv2.threshold(fgMask, 244, 255, cv2.THRESH_BINARY)[1]
    
    # Apply some morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Update existing trackers
    objects_to_delete = []
    for object_id, (tracker, last_pos) in trackers.items():
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {object_id}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            trackers[object_id] = (tracker, box)  # Update last known position
        else:
            objects_to_delete.append(object_id)
    
    # Remove failed trackers
    for object_id in objects_to_delete:
        del trackers[object_id]
    
    # Process new detections
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this object overlaps with any existing tracker
            overlap = False
            for _, (_, last_pos) in trackers.items():
                tx, ty, tw, th = [int(v) for v in last_pos]
                if (x < tx + tw and x + w > tx and y < ty + th and y + h > ty):
                    overlap = True
                    break
            
            if not overlap:
                # Create a new tracker for this object
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))
                trackers[next_object_id] = (tracker, (x, y, w, h))
                next_object_id += 1
    
    # Display the result
    cv2.imshow('Frame', frame)
    # cv2.imshow('Mask', thresh)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()