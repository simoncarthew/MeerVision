import cv2
import torch
import os
from ultralytics import YOLO

class Yolo8:

    def __init__(self, yolo_path = 'ObjectDetection/Yolo8', model_load = "pre"):
        # Load selected model
        if model_load == "new":
            model_size = input("Enter model size (n/m/s/l): ")
            self.model = YOLO(yolo_path + "/yolov8" + model_size + ".yaml")
        elif model_load == "tune":
            model_size = input("Enter model size (n/m/s/l): ")
            self.model = YOLO(yolo_path + "/yolov8" + model_size + ".pt")
        elif model_load == "latest":
            # all directories
            all_directories = [d for d in os.listdir(yolo_path) if os.path.isdir(os.path.join(yolo_path, d))]

            # train directories
            train_directories = [d for d in all_directories if d.startswith('train')]

            # sort trained numberically
            train_directories_sorted = sorted(train_directories, key=lambda x: int(x[5:]) if x[5:].isdigit() else 0)

            # get the latest directory
            latest_directory = train_directories_sorted[-1] if train_directories_sorted else None

            self.model = YOLO(yolo_path + "/" + latest_directory + "/weights/best.pt")
        elif model_load == "custom":
            model_path = input("Please paste model path: ")
            self.model = YOLO(model_path)
    
    def train(self, batch = 32, imgsz = 640, lr = 0.01, optimizer = 'SGD', epochs = 50, dataset_path="Data/MeerDown/yolo/dataset.yaml", save_path = 'ObjectDetection/Yolo8', augment = False):
        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # set augmenting variables
        results = self.model.train(
            device=device,
            lr0=lr,
            project=save_path,
            verbose=True,
            optimizer=optimizer,
            data=dataset_path,  # Path to the dataset YAML file
            epochs=epochs,  # Number of training epochs
            batch=batch,  # Batch size
            imgsz=imgsz,  # Image size
            augment=augment,  # Enable data augmentation
            hsv_h=0.015,  # Hue adjustment
            hsv_s=0.7,    # Saturation adjustment
            hsv_v=0.4,    # Value (brightness) adjustment
            degrees=10.0,  # Rotation in degrees (adjusted to a non-zero value)
            translate=0.1,  # Translation fraction
            scale=0.5,  # Scaling factor
            shear=2.0,  # Shear angle in degrees (adjusted to a non-zero value)
            perspective=0.001,  # Perspective transformation (adjusted to a non-zero value)
            flipud=0.1,  # Probability of flipping vertically (adjusted to a non-zero value)
            fliplr=0.5,  # Probability of flipping horizontally
            mosaic=1.0,  # Mosaic augmentation
            mixup=0.2,  # MixUp augmentation (adjusted to a non-zero value)
            copy_paste=0.2,  # Copy-Paste augmentation (adjusted to a non-zero value)
        )

        return results
    
    def evaluate_model(self, dataset_path="Data/MeerDown/yolo/dataset.yaml", split="test"):
        # Evaluate the model on the specified dataset split (e.g., "test")
        metrics = self.model.val(data=dataset_path, split=split)
        
        # Extract mAP@0.5 score
        map_50 = metrics['metrics/mAP_0.5']
        print(f"mAP@0.5: {map_50:.4f}")
        
        return metrics

    def process_video(self, video_path, thresh=0.3, save_path=None):
        # Load the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video was loaded correctly
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        # Get the width, height, and frames per second of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize the video writer if save_path is not None
        if save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        # Process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict bounding boxes on the current frame
            results = self.model(frame)

            # Draw bounding boxes on the frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0]  # Confidence score
                    if conf > thresh:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                        cls = int(box.cls[0])  # Class label
                        label = self.model.names[cls]

                        # Draw the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the frame to the output video if saving is enabled
            if save_path is not None:
                out.write(frame)

            # Display the frame with bounding boxes
            cv2.imshow('YOLOv8 Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and writer objects, and close display windows
        cap.release()
        if save_path is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    # would you like to train ?
    train = False
    if input("Would you like to train a model (y/n)? ") == "y":
        train = True

    # would you like to train new or start fresh?
    train_select = ""
    while (train_select == "" and train == True):
        train_select = input("Select train mode (new/tune/latest)? ")
        if train_select not in ["new", "latest", "tune"]:
            train_select = ""

    # select latest if no training
    if train == False:
        if input("Would you like to use custome model (y/n)? ") == "y":
            train_select = "custom"
        else:
            train_select = "latest"

    # load the model
    yolo8 = Yolo8(model_load=train_select)

    # would you like to show test video
    video = False
    if input("Would you like to see test video(y/n)? ") == "y":
        video = True

    # train model
    if train == True:
        yolo8.train(dataset_path='Data/Formated/yolo/dataset.yaml',epochs=7, batch = 16, lr=0.01, augment = True)

    # test the trained model
    if video:
        video_path = input("Please paste video path: ")
        thresh = 0.3
        if video_path =="":
            yolo8.process_video("Data/MeerDown/origin/Unannotated_videos/22-11-07_C3_10.mp4",thresh=thresh)
        else:
            yolo8.process_video(video_path,thresh=thresh, save_path="ObjectDetection/Yolo8/output/output.mp4")