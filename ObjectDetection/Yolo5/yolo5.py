import cv2
import torch
import os

class Yolo5:

    def __init__(self, yolo_path='ObjectDetection/Yolo5', model_load="pre"):
        # Load selected model
        if model_load == "new":
            model_size = input("Enter model size (s/m/l/x): ")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path + "/yolov5" + model_size + ".yaml")
        elif model_load == "tune":
            model_size = input("Enter model size (s/m/l/x): ")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path + "/yolov5" + model_size + ".pt")
        elif model_load == "latest":
            # all directories
            all_directories = [d for d in os.listdir(yolo_path) if os.path.isdir(os.path.join(yolo_path, d))]

            # train directories
            train_directories = [d for d in all_directories if d.startswith('train')]

            # sort trained directories numerically
            train_directories_sorted = sorted(train_directories, key=lambda x: int(x[5:]) if x[5:].isdigit() else 0)

            # get the latest directory
            latest_directory = train_directories_sorted[-1] if train_directories_sorted else None

            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path + "/" + latest_directory + "/weights/best.pt")
        elif model_load == "custom":
            model_path = input("Please paste model path: ")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    
    def train(self, batch=16, imgsz=640, lr=0.02, optimizer='AdamW', epochs=50, dataset_path="Data/MeerDown/yolo/dataset.yaml", save_path='ObjectDetection/Yolo5', augment=False):
        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # set augmenting variables
        if augment:
            self.model.train(data=dataset_path, batch_size=batch, epochs=epochs, imgsz=imgsz, project=save_path,
                             save=True, optimizer=optimizer, lr0=lr, device=device,
                             augment=True)  # Augment=True applies default augmentations in YOLOv5
        else:
            self.model.train(data=dataset_path, batch_size=batch, epochs=epochs, imgsz=imgsz, project=save_path,
                             save=True, optimizer=optimizer, lr0=lr, device=device)

    def process_video(self, video_path, thresh=0.3):
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
            for result in results.xyxy[0]:  # result.xyxy[0] contains predictions for the first image
                conf = result[4]  # Confidence score
                if conf > thresh:
                    x1, y1, x2, y2 = map(int, result[:4])  # Bounding box coordinates
                    
                    cls = int(result[5])  # Class label
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

    # would you like to train?
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
    if not train:
        if input("Would you like to use custom model (y/n)? ") == "y":
            train_select = "custom"
        else:
            train_select = "latest"

    # load the model
    yolo5 = Yolo5(model_load=train_select)

    # would you like to show test video
    video = False
    if input("Would you like to see test video(y/n)? ") == "y":
        video = True

    # train model
    if train:
        yolo5.train(dataset_path='Data/MeerDown/yolo/yolo_val_mix_half/dataset.yaml', epochs=7, batch=4, lr=0.02, augment=False)

    # test the trained model
    if video:
        video_path = input("Please paste video path: ")
        thresh = 0.1
        if video_path == "":
            yolo5.process_video("Data/MeerDown/origin/Unannotated_videos/22-11-07_C3_10.mp4", thresh=thresh)
        else:
            yolo5.process_video(video_path, thresh=thresh)
