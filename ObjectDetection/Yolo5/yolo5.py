# BLOVK WARNINGS
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# IMPORTS
import torch
import cv2
import time
import glob
import os
import sys
from yolov5 import train, detect, val  # Make sure YOLOv5 repository is in your PYTHONPATH

# IMPORT EVAL
eval_path = os.path.join("ObjectDetection")
sys.path.append(eval_path)
from Evaluate import EvaluateModel

class Yolo5:
    def __init__(self, model_size='s', model_path = None, pretrained = True, device=None):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_size = model_size
        if model_path is not None:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            model_size = "yolov5" + model_size
            self.model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=pretrained).to(self.device)

    def train(self, data_path, lr = 0.01, epochs=30, batch_size=16, img_sz=640, freeze = 0, optimizer = 'SGD', augment = True, save_path = os.path.join("ObjectDetection","Yolo5")):
        if freeze != 0:
            freeze=list(range(freeze))
        else:
            freeze = [0]

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
            name = "train",
            freeze=freeze,
            weights = "yolov5" + self.model_size + ".pt"
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
    
    def native_evaluate(self, data_path, model_path, save_path, task="test"):
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
    
    def evaluate(self, yolo_path, img_width=640, img_height=640):
        pred_detections = self.test_detect(yolo_path=yolo_path)
        eval = EvaluateModel(yolo_path,pred_detections,img_width,img_height)
        results = eval.run_evaluation()
        results['inference'] = self.inference_time(os.path.join(yolo_path,"images","test"))
        return results

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

        return avg_inf

    def sgl_detect(self, image_path, show=False, conf_thresh=0, format="yolo"):
        img = cv2.imread(image_path)  # Read the image
        results = self.model(img)  # Perform detection

        # Extract results
        detected_boxes = []
        for result in results.xyxy[0]:  # results.xyxy[0] gives the detections
            x1, y1, x2, y2, conf, cls = result.tolist()
            if conf >= conf_thresh and int(cls) == 0:
                if format == "std":
                    box = [int(x1), int(y1), int(x2), int(y2)]
                elif format == "yolo":
                    h, w = img.shape[:2]
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
                    'bbox': box,  # Bounding box coordinates
                    'score': float(conf),  # Confidence score
                    'class': int(cls)  # Class label
                })

        if show:
            self.draw_detection(detected_boxes, img, thresh=conf_thresh, format=format)

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
                {"id": 0, "name": "meerkat"}  # Example class, replace as needed
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
                # You can use any library to get image dimensions. Here we assume OpenCV
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
                    "bbox": detection["bbox"],
                    "score": detection["score"],  # Detection confidence score
                    "area": detection["bbox"][2] * detection["bbox"][3],  # bbox_width * bbox_height
                    "iscrowd": 0,  # COCO requires this field, setting it to 0 (non-crowd)
                    "segmentation": []  # If no segmentation, leave it empty
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1
        return coco_format

if __name__ == "__main__":

    # Example usage
    # model_path = "ObjectDetection/Yolo5/best.pt"
    # model_path = '/home/meerkat/MeerVision/Control/Models/yolo5.pt'
    # model_path = "ObjectDetection/Training/Results/hyper_tune/results0/models/model_0/weights/best.pt"
    # model_path = "ObjectDetection/Yolo5/train/weights/best.pt"
    # model_path = "ObjectDetection/Training/Results/yolo5_first_test/models/model_0/weights/best.pt"
    # print("Loading Previous Model")
    # yolo = Yolo5(model_path=model_path)
    # yolo.detect_video("Data/YoutubeCameraTrap/At the meerkat burrow.mp4")
    # print("Previous Model Loaded")
    # print("Loading new Model")
    yolo = Yolo5(model_size='m')
    # print("New Model Loaded")
    # jpg_files = glob.glob(os.path.join("Data/Formated/yolo/images/test", '*.jpg'))
    # for file in jpg_files:
    #     print(yolo.sgl_detect(image_path=file, show=True))
    # print("Starting Training")
    yolo.train(data_path='Data/Formated/yolo/dataset.yaml',epochs=5,batch_size=4,freeze = 10)
    # print("Finnished Training")
    # print(yolo.evaluate_model("Data/Formated/yolo/dataset.yaml",model_path,save_path='ObjectDetection/Yolo5/testing'))
    # print(yolo.cust_evaluate(yolo_path="Data/Formated/yolo"))
    # detections = yolo.sgl_detect("Data/Formated/yolo/images/test/At the meerkat burrow_26.jpg", show = False,format="std")
    # print(detections)
    # yolo.draw_detection(detected_boxes=detections,img=cv2.imread("Data/Formated/yolo/images/test/At the meerkat burrow_26.jpg"),format="std",show=False,save_path="test.jpg")
    # print(yolo.native_evaluate(os.path.join('Data','Formated','yolo','dataset.yaml'), model_path, save_path='ObjectDetection/Yolo5/testing', task="test"))
    # print(yolo.evaluate(os.path.join('Data','Formated','yolo')))

    # yolo = Yolo5(model_size='s')
    # yolo.train(data_path='/scratch/crtsim008/Formated/yolo/dataset.yaml',epochs=20,batch_size=32)

    # yolo = Yolo5(model_size='m')
    # yolo.train(data_path='/scratch/crtsim008/Formated/yolo/dataset.yaml',epochs=20,batch_size=32)

    # yolo = Yolo5(model_size='l')
    # yolo.train(data_path='/scratch/crtsim008/Formated/yolo/dataset.yaml',epochs=20,batch_size=32)