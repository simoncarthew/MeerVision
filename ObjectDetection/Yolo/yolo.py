# IMPORTS
import cv2
import torch
import os
import glob
import time
import sys
import math
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

# IMPORT EVAL
eval_path = os.path.join("ObjectDetection")
sys.path.append(eval_path)
from Evaluate import EvaluateModel

class Yolo:
    
    def __init__(self, model_size='s', model_path = None, pretrained = True, device=None, yolo_path = os.path.join("ObjectDetection","Yolo")):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is not None:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(os.path.join(yolo_path, "yolov" + model_size + ".pt"))

    def train(self, batch = 32, freeze = 0, img_sz = 640, lr = 0.01, optimizer = 'SGD', epochs = 50, dataset_path=os.path.join("Data","Formated","yolo","dataset.yaml"), save_path = os.path.join('ObjectDetection','Yolo'), augment = False):

        # set augmenting variables
        results = self.model.train(
            device=self.device,
            lr0=lr,
            project=save_path,
            verbose=True,
            optimizer=optimizer,
            data=dataset_path,  
            epochs=epochs, 
            batch=batch,  
            imgsz=img_sz,  
            augment=augment,
            freeze=freeze,
            single_cls=True
        ) 

        return results
    
    def tune(self, iterations = 5, batch = 32, freeze = 0, img_sz = 640, lr = 0.01, optimizer = 'SGD', epochs = 1, dataset_path=os.path.join("Data","Formated","yolo","dataset.yaml"), save_path = os.path.join('ObjectDetection','Yolo'), augment = False):

        # set augmenting variables
        results = self.model.tune(
            device=self.device,
            lr0=lr,
            project=save_path,
            verbose=True,
            optimizer=optimizer,
            data=dataset_path,  
            epochs=epochs, 
            batch=batch,  
            imgsz=img_sz,  
            augment=augment,
            freeze=freeze,
            iterations=iterations
        ) 

        return results

    def native_evaluate(self, dataset_path=os.path.join("Data","Formated","yolo","dataset.yaml"), split="test"):
        results = self.model.val(data=dataset_path, split=split)
        return results.results_dict

    def evaluate(self, yolo_path, img_width=640, img_height=640):
        pred_detections = self.test_detect(yolo_path=yolo_path)
        eval = EvaluateModel(yolo_path,pred_detections,img_width,img_height)
        results = eval.run_evaluation()
        results['inference'] = self.inference_time(image_folder=os.path.join(yolo_path,"images","test"))
        return results

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

    def inference_time(self, image_folder):
        img_files = glob.glob(os.path.join(image_folder, "*.jpg"))
        inf_times = []

        for img_file in tqdm(img_files):
            # time inference
            img = cv2.imread(img_file)
            start_time = time.time()
            results = self.model(img)
            end_time = time.time()
            inf_times.append(end_time - start_time)

        # calculate averages
        avg_inf = sum(inf_times[1:])/len(inf_times[1:])

        return avg_inf

    def to_pi(self):
        self.model.export(format="ncnn")

    def sgl_detect(self, image_path, show=False, conf_thresh=0, format="yolo", save_path=None, crop_save_path= None):
        img = cv2.imread(image_path)  # Read the image
        if img is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        
        results = self.model(img)  # Perform detection

        # Ensure crop save path is defined and create it if not present
        if crop_save_path and not os.path.exists(crop_save_path):
            os.makedirs(crop_save_path)

        # Extract results
        detected_boxes = []
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes in xyxy format
            confidences = result.boxes.conf  # Get confidence scores
            classes = result.boxes.cls  # Get class labels
            id = 0

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box.tolist()

                # Ensure the confidence score is valid
                if conf is None or any(math.isnan(val) for val in [x1, y1, x2, y2]):
                    continue  # Skip this bounding box if invalid

                if conf >= conf_thresh and int(cls) == 0:  # Class 0 condition
                    h, w = img.shape[:2]
                    if w == 0 or h == 0:
                        raise ValueError(f"Image dimensions are invalid: width={w}, height={h}")
                    
                    # YOLO format bounding box conversion
                    if format == "std":
                        box = [int(x1), int(y1), int(x2), int(y2)]
                    elif format == "yolo":
                        x_center = (x1 + x2) / 2 / w
                        y_center = (y1 + y2) / 2 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        box = [x_center, y_center, width, height]
                    elif format == "coco":
                        box = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    else:
                        print(f"{format} is not a supported box format.")
                        exit()

                    detected_boxes.append({
                        'image_id': os.path.basename(image_path),
                        'bbox': box,
                        'score': float(conf),
                        'class': int(cls)
                    })

                    # Crop the detected region and save if crop_save_path is specified
                    if crop_save_path:
                        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                        crop_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_crop_{id}.jpg"
                        cv2.imwrite(os.path.join(crop_save_path, crop_filename), cropped_img)
                    id+=1

        if save_path:
            self.draw_detection(detected_boxes, img, thresh=conf_thresh, format=format,save_path=save_path, show=show)

        return detected_boxes

    def draw_detection(self, detected_boxes, img, thresh=0, format="yolo", show=True, save_path=None):
        for detected in detected_boxes:
            conf = detected['score']
            cls = detected['class']
            label = self.model.names[cls]  # Get class name from model
            
            if conf > thresh:
                if format == "std":
                    x1, y1, x2, y2 = detected['bbox']
                elif format == "yolo":
                    x_center, y_center, width, height = detected['bbox']
                    h, w = img.shape[:2]
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                elif format == "coco":
                    x1, y1, width, height = detected['bbox']
                    x2 = x1 + width
                    y2 = y1 + height
                else:
                    print(f"{format} is not a supported box format.")
                    continue

                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with detections
        if show:
            cv2.imshow('Detection', img)
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty('Detection', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()

        # save the image if requested
        if save_path:
            cv2.imwrite(save_path, img)

    def test_detect(self, yolo_path, conf_thresh=0):
        # Get image files
        img_files = glob.glob(os.path.join(yolo_path, "images", "test", '*.jpg'))
        
        # Initialize COCO structure
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "meerkat"}
            ]
        }

        annotation_id = 0

        # Iterate over images
        for image_id, img_file in enumerate(img_files):
            # Add image information to COCO format
            img_name = os.path.basename(img_file)
            img_info = {
                "id": image_id,
                "file_name": img_name,
                "height": cv2.imread(img_file).shape[0],
                "width": cv2.imread(img_file).shape[1]
            }
            coco_format["images"].append(img_info)

            # Get detections from the image
            detections = self.sgl_detect(img_file, conf_thresh=conf_thresh,format="coco")  # Adjust conf_thresh as needed

            # Add detections to COCO format
            for detection in detections:
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": detection["class"],  # Class ID, assuming 0 for meerkats
                    "bbox": detection["bbox"],  # YOLO bbox already in [x_min, y_min, width, height]
                    "score": detection["score"],  # Detection confidence score
                    "area": detection["bbox"][2] * detection["bbox"][3],  # bbox_width * bbox_height
                    "iscrowd": 0,  # COCO requires this field, setting it to 0 (non-crowd)
                    "segmentation": []  # If no segmentation, leave it empty
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

        return coco_format

    def look_at_layers(self):

        # Access the model's convolutional layers
        # Depending on the model architecture, we may need to navigate to the backbone
        conv_layers = [layer for layer in self.model.modules() if isinstance(layer, torch.nn.Conv2d)]

        if not conv_layers:
            raise ValueError("No convolutional layers found in the model.")
        
        # Visualize kernels of the first convolutional layer
        kernels = conv_layers[0].weight.data.cpu()  # Get the weights of the first conv layer

        # Plot the kernels
        num_kernels = kernels.shape[0]
        plt.figure(figsize=(12, 12))
        for i in range(num_kernels):
            plt.subplot(8, 8, i + 1)
            plt.imshow(kernels[i].squeeze(), cmap='gray')
            plt.axis('off')

        plt.show()

if __name__ == "__main__":
    # # initialize model
    # model_path = "ObjectDetection/Training/Results/hyper_tune/results0/models/model_0/weights/best.pt"
    # model_path = "ObjectDetection/Yolo/hpc.pt"
    model_path = "ObjectDetection/Training/Results/merged_sz_results/models/model_16.pt"
    # yolo = Yolo(model_size="5nu")
    yolo = Yolo(model_path=model_path)
    # yolo.look_at_layers()
    
    # training
    # yolo.train(batch = 4)

    # test tuning
    # yolo.tune(batch = 8, dataset_path = "Data/Formated/test_dataset/dataset.yaml")

    # convert to pi
    # yolo.to_pi()

    # detct image
    # image_path = "Data/ReportImages/test_0_undetected.jpg"
    # yolo.sgl_detect(image_path,show=False,save_path="Data/ReportImages/test_0_detected.jpg",conf_thresh=0.5)
    # image_path = "Data/ReportImages/test_1_undetected.jpg"
    # yolo.sgl_detect(image_path,show=False,save_path="Data/ReportImages/test_1_detected.jpg",conf_thresh=0.5)
    # image_path = "Data/ReportImages/test_2_undetected.jpg"
    # yolo.sgl_detect(image_path,show=False,save_path="Data/ReportImages/test_2_detected.jpg",conf_thresh=0.5)
    # image_path = "Data/ReportImages/test_3_undetected.jpg"
    # print(yolo.sgl_detect(image_path,show=False,save_path="Data/ReportImages/test_3_detected.jpg",conf_thresh=0.5, crop_save_path="Data/ReportImages"))

    # image_path = "Data/ReportImages/MedLight/2024_10_12_12_0_51.jpg"
    # yolo.sgl_detect(image_path,show=True,save_path=None)

    # test video
    # video_path = "Data/YoutubeCameraTrap/istockphoto-1990464825-640_adpp_is.mp4"
    # video_path = "Data/YoutubeCameraTrap/At the meerkat burrow.mp4"
    # video_path = "Data/YoutubeCameraTrap/istockphoto-892591066-640_adpp_is.mp4"
    # video_path = "Data/YoutubeCameraTrap/istockphoto-2062853095-640_adpp_is.mp4"
    video_path = "Data/Staceys/VID20230113185107.mp4"
    yolo.process_video(video_path,thresh=0.4)

    # evaluating
    # print(yolo.native_evaluate())