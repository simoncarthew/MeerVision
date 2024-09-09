import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2
import time
from pathlib import Path
from yolov5 import train, detect  # Make sure YOLOv5 repository is in your PYTHONPATH

class Yolo5:
    def __init__(self, model_size='s', device=None):
        model_size = "yolov5" + model_size
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=True).to(self.device)
    
    def freeze_layers(self, freeze):
        # Print the layers that will be frozen
        for k, v in self.model.named_parameters():
            if any(f"model.{x}." in k for x in range(freeze)):
                v.requires_grad = False
            else:
                v.requires_grad = True

    def train(self, data_path, epochs=30, batch_size=16, img_sz=640, freeze = 10, optimizer = 'SGD', augment = True, save_path = "ObjectDetection/Yolo5"):
        self.freeze_layers(freeze)

        train.run(
            data=data_path,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_sz,
            device=self.device,
            project = save_path,
            optimizer = optimizer,
            augment=augment
            name = "train"
        )
    
    def detect_video(self, video_path, output_path=None, conf_thres=0.25):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object to save output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to a format YOLOv5 expects (BGR to RGB, then resize)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(img, size=640)  # Run inference

            # Draw bounding boxes and labels on the frame
            annotated_frame = results.render()[0]  # results.render() returns a list of images (in BGR format)
            
            # Display the annotated frame
            cv2.imshow('YOLOv5 Detection', annotated_frame)

            # Write the frame to the output video
            if output_path:
                out.write(annotated_frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    # Example usage
    yolo = Yolo5(model_size='s')

    # Train the model on a custom dataset
    yolo.train(data_path='Data/Formated/yolo/dataset.yaml', epochs=1)

    # Detect and annotate objects in a video
    # yolo.detect_video(video_path='input_video.mp4', output_path='output_video.mp4')


