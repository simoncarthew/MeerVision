import cv2
import torch
import os
import glob
from ultralytics import YOLO

class Yolo8:
    
    def __init__(self, model_size='s', model_path = None, pretrained = True, device=None, yolo_path = os.path.join("ObjectDetection","Yolo8")):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is not None:
            self.model = YOLO(model_path)
        else:
            if pretrained:
                self.model = YOLO(yolo_path + "/yolov8" + model_size + ".pt")
            else:
                self.model = YOLO(yolo_path + "/yolov8" + model_size + ".yaml")
        
    
    def freeze_layers(self, freeze):
        freeze = [f"model.{x}." for x in range(freeze)]
        for k, v in self.model.named_parameters():
            v.requires_grad = True 
            if any(x in k for x in freeze):
                v.requires_grad = False

    def train(self, batch = 32, freeze = 0, img_sz = 640, lr = 0.01, optimizer = 'SGD', epochs = 50, dataset_path="Data/MeerDown/yolo/dataset.yaml", save_path = 'ObjectDetection/Yolo8', augment = False):
        self.freeze_layers(freeze)

        # set augmenting variables
        results = self.model.train(
            device=self.device,
            lr0=lr,
            project=save_path,
            verbose=True,
            optimizer=optimizer,
            data=dataset_path,  # Path to the dataset YAML file
            epochs=epochs,  # Number of training epochs
            batch=batch,  # Batch size
            imgsz=img_sz,  # Image size
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
    
    def evaluate_model(self, dataset_path="Data/Formated/yolo/dataset.yaml", split="test"):
        results = self.model.val(data=dataset_path, split=split)
        return results.results_dict

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

    def inference_time(self, yolo_path):
        test_path = os.path.join(yolo_path,"images","test")
        img_files = glob.glob(f"{test_path}/*.jpg")
        pre_times = []
        inf_times = []
        post_times = []

        for img_file in img_files:
            # time inference
            img = cv2.imread(img_file)
            results = self.model(img)

            # append times
            pre_times.append(results[0].speed["preprocess"])
            post_times.append(results[0].speed["postprocess"])
            inf_times.append(results[0].speed["inference"])

        # calculate averages
        avg_pre = sum(pre_times)/len(pre_times)
        avg_post = sum(post_times)/len(post_times)
        avg_inf = sum(inf_times)/len(inf_times)

        # create output dictionary
        output = {"avg_inf":avg_inf,"avg_post":avg_post,"avg_pre":avg_pre}

        return output
    
    def detect(self, image_path, show = False, conf_thresh=0):
        img = cv2.imread(image_path)  # raed image
        results = self.model(img)  # detect
        detected_boxes = [] 

        # Iterate through results and filter based on the confidence threshold
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0].item()
                if conf >= conf_thresh: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                    detected_boxes.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,  
                        'class': int(box.cls[0])
                    })

        if show: self.draw_detection(detected_boxes,img)

        return detected_boxes

    def draw_detection(self, detected_boxes, img, thresh = 0):
        for detected in detected_boxes:
            x1, y1, x2, y2 = detected['box']
            conf = detected['confidence']
            cls = detected['class']
            label = self.model.names[cls] 

            if conf > thresh:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 Detection', img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('YOLOv8 Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":

    yolo = Yolo8(model_path="ObjectDetection/Yolo8/colab/train13/weights/best.pt")
    # print(yolo.inference_time(yolo_path="Data/Formated/yolo"))
    detections = yolo.detect(image_path="Data/Formated/yolo/images/test/Suricata_Desmarest_86.jpg")