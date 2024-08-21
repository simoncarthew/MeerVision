import cv2
import torch
import os
from ultralytics import YOLO

class Yolo8:

    def __init__(self, yolo_path = 'ObjectDetection/Yolo8', model_load = "pre"):
        # Load selected model
        if model_load == "new":
            self.model = YOLO(yolo_path + '/yolov8n.yaml')
        elif model_load == "tune":
            self.model = YOLO(yolo_path + '/yolov8n.pt')
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
    
    def train(self, batch = 16, imgsz = 640, lr = 0.01, optimizer = 'AdamW', epochs = 50, dataset_path="Data/MeerDown/yolo/dataset.yaml", save_path = 'ObjectDetection/Yolo8', augment = False):
        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # set augmenting variables
        if augment == True:
            results = self.model.train(
                # set general parameters
                data=dataset_path,
                batch=batch,
                epochs=epochs,
                imgsz=imgsz,
                project=save_path,
                save=True,          
                save_period=-1, 
                verbose=True,
                lr0=lr,
                optimizer=optimizer,
                amp=False,
                device=device,

                # data augmentation
                mosaic=0.5,         # Mosaic augmentation probability
                mixup=0.3,          # Mixup augmentation probability
                hsv_h=0.02,         # HSV hue augmentation (default is 0.015)
                hsv_s=0.7,          # HSV saturation augmentation (default is 0.7)
                hsv_v=0.5,          # HSV value augmentation (default is 0.4)
                flipud=0.0,         # Vertical flip probability (0.0 = disabled)
                fliplr=0.5,         # Horizontal flip probability (default is 0.5)
                shear=0.2,          # Shear augmentation magnitude (default is 0.0)
                perspective=0.002,  # Perspective augmentation magnitude (default is 0.0)
                scale=0.5,          # Scale augmentation magnitude (default is 0.5)
                translate=0.2,      # Image translation (+/- fraction)
                degrees=7.5,        # Image rotation (+/- degrees)
                erasing=0.5,        # Probability of random erasing during classification training
                auto_augment='autoaugment'  # Auto augmentation policy
            )
        else:
            results = self.model.train(
                # set general parameters
                data=dataset_path,
                batch=batch,
                epochs=epochs,
                imgsz=imgsz,
                project=save_path,
                save=True,          
                save_period=-1, 
                verbose=True,
                lr0=lr,
                optimizer=optimizer,
                amp=False,
                device=device
            )

        return results

    def process_video(self, video_path, thresh = 0.3):
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

            # Display the frame with bounding boxes
            cv2.imshow('YOLOv8 Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close display windows
        cap.release()
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
        train_select = "latest"

    # load the model
    yolo8 = Yolo8(model_load=train_select)

    # train model
    if train == True:
        yolo8.train(dataset_path='Data/MeerDown/yolo_merged/dataset.yaml',epochs=50, augment = False)

    # test the trained model
    yolo8.process_video("Data/MeerDown/origin/Unannotated_videos/22-11-07_C3_10.mp4",thresh=0.1)