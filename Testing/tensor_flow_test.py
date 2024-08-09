import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Load the pre-trained model
model = hub.load('https://tfhub.dev/google/openimages_v4/inception_v2/1')

def run_inference_for_single_image(image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)
    return output_dict

# Load the video footage
cap = cv2.VideoCapture('/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Testing/Vision/meerkat_test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output_dict = run_inference_for_single_image(frame)

    # Process and display the output dictionary (implement based on the model output format)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
