import cv2
import torch
from yolov5 import YOLOv5

class Yolo5:

    def __init__(self, model_load = "pre"):
        
        # Load the selected model
        if model_load == "pre":
            self.model = YOLOv5('ObjectDetection/Yolo5/yolov5s.pt')
        elif model_load == "tuned":
            self.model = YOLOv5('ObjectDetection/Yolo5/runs/train/weights/best.pt')
    
    def train(self, dataset_path="Data/MeerDown/yolo/dataset.yaml", save_path = 'ObjectDetection/Yolo5'):
        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Training on device: {device}")
        
        # Start training
        print("Starting training...")
        self.model.train(
            data=dataset_path,
            epochs=10,
            imgsz=640,
            project=save_path,
            device=device,
            save=True,
            save_period=-1,
            verbose=True
        )

        # Training results are saved in the project directory
        print("Training completed. Check the project directory for results.")

    def process_video(self, video_path):
        # Load the video file
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
            results = self.model(frame)

            # Draw bounding boxes on the frame
            for result in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = map(int, result[:6])
                label = self.model.names[cls]

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame with bounding boxes
            cv2.imshow('YOLOv5 Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close display windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load pretrained model
    yolo5 = Yolo5()

    # Train model
    yolo5.train()

    # Test the trained model
    yolo5.process_video("Data/MeerDown/origin/Annotated_videos/area_2/22-10-20_C3_04.mp4")
