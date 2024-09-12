import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2
import time
import glob
import os
from yolov5 import train, detect, val  # Make sure YOLOv5 repository is in your PYTHONPATH

class Yolo5:
    def __init__(self, model_size='s', model_path = None, pretrained = True, device=None):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        if model_path is not None:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            model_size = "yolov5" + model_size
            self.model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=pretrained).to(self.device)
    
    def freeze_layers(self, freeze):
        # Print the layers that will be frozen
        for k, v in self.model.named_parameters():
            if any(f"model.{x}." in k for x in range(freeze)):
                v.requires_grad = False
            else:
                v.requires_grad = True

    def train(self, data_path, lr = 0.01, epochs=30, batch_size=16, img_sz=640, freeze = 10, optimizer = 'SGD', augment = True, save_path = "ObjectDetection/Yolo5"):
        self.freeze_layers(freeze)

        train.run(
            data=data_path,
            lr0 = lr,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_sz,
            device=self.device,
            project = save_path,
            optimizer = optimizer,
            augment=augment,
            name = "train"
        )
    
    def detect_video(self, video_path, output_path=None, conf_thres=0.25):
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Set up video writer (optional)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            results = self.model(frame)

            # Render results on the frame
            rendered_frame = results.render()[0]  # returns a list of images

            # Display the frame
            cv2.imshow('YOLOv5 Detection', rendered_frame)

            # Write the frame to the output video
            if output_path: out.write(rendered_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def evaluate_model(self, data_path, model_path, save_path, task="test"):
        results = val.run(
            data=data_path,
            weights=model_path,
            device=self.device,
            project=save_path,
            task=task
        )
        
        # Unpack the results
        mp, mr, map50, map, *losses = results[0]  # Unpack the first tuple containing metrics
        maps = results[1]  # Class-specific mAPs
        times = results[2]  # Inference time and NMS time
        
        # Create the results dictionary
        results_dict = {
            'metrics/precision(B)': mp,
            'metrics/recall(B)': mr,
            'metrics/mAP50(B)': map50,
            'metrics/mAP50-95(B)': map,
            'loss/box_loss': losses[0],  # Assuming losses are [box_loss, obj_loss, cls_loss]
            'loss/obj_loss': losses[1],
            'loss/cls_loss': losses[2],
            'class_specific_mAPs': maps.tolist(),  # Convert to list if needed
            'times/inference_time': times[0],
            'times/nms_time': times[1]
        }
        
        return results_dict
    
    def inference_time(self, image_folder):
        img_files = glob.glob(f"{image_folder}/*.jpg")
        inf_times = []

        for img_file in img_files:
            img = cv2.imread(img_file)
            start_time = time.time()
            results = self.model(img)
            end_time = time.time()
            inf_times.append(end_time - start_time)

        avg_inf = sum(inf_times) / len(inf_times)

        output = {"avg_inf": avg_inf}

        return output

    
    def sgl_detect(self, image_path, show=False, conf_thresh=0):
        img = cv2.imread(image_path)  # Read the image
        results = self.model(img)  # Perform detection

        # Extract results
        detected_boxes = []
        for result in results.xyxy[0]:  # results.xyxy[0] gives the detections
            x1, y1, x2, y2, conf, cls = result.tolist()
            if conf >= conf_thresh and int(cls) == 0:
                detected_boxes.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),  # Bounding box coordinates
                    'confidence': float(conf),  # Confidence score
                    'class': int(cls)  # Class label
                })

        if show:
            self.draw_detection(detected_boxes, img)

        return detected_boxes
    
    def test_detect(self,yolo_path):
        # get image files
        img_files = glob.glob(os.path.join(yolo_path, "images", "test", '*.jpg'))
        
        # initialise all detections
        all_detections = {}

        # iterate over images
        for img_file in img_files:
            detection = self.sgl_detect(img_file,show=False)
            all_detections[os.path.basename(img_file)] = detection
        
        return all_detections

    def draw_detection(self, detected_boxes, img, thresh=0):
        for detected in detected_boxes:
            x1, y1, x2, y2 = detected['box']
            conf = detected['confidence']
            cls = detected['class']
            label = self.model.names[cls]  # Get class name from model

            if conf > thresh:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLOv5 Detection', img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('YOLOv5 Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

        
if __name__ == "__main__":

    # Example usage
    model_path = "ObjectDetection/Yolo5/train/weights/best.pt"
    # model_path = "ObjectDetection/Yolo5/md_v5b.0.0.pt"
    yolo = Yolo5(model_size='s',model_path=model_path)
    # jpg_files = glob.glob(os.path.join("Data/Formated/yolo/images/test", '*.jpg'))
    # for file in jpg_files:
    #     print(yolo.detect(image_path=file, show=True))
    print(yolo.test_detect(yolo_path='Data/Formated/yolo'))
    # print(yolo.evaluate_model("Data/Formated/yolo/dataset_mega.yaml",model_path,save_path='ObjectDetection/Yolo5/testing'))