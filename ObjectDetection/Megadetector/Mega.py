import cv2
import torch
import os
import sys
import glob
import time

# import evaluation
eval_path = os.path.join("ObjectDetection")
sys.path.append(eval_path)
from Evaluate import EvaluateModel

# import yolo5
yolo_path = os.path.join("ObjectDetection","Yolo5")
sys.path.append(yolo_path)
from yolo5 import Yolo5

# import CNN
cnn_path = os.path.join("Classification")
sys.path.append(cnn_path)
from CNNS import CNNS

# model paths
MODEL_A_PATH = os.path.join("ObjectDetection","Megadetector","md_v5a.0.0.pt")
MODEL_B_PATH = os.path.join("ObjectDetection","Megadetector","md_v5b.0.0.pt")

class Mega:
    def __init__(self, version, class_path, class_name, img_size=(640, 640)):

        # load selected mega version
        if version == "a":
            self.yolo = Yolo5(model_path = MODEL_A_PATH)
        elif version == "b":
            self.yolo = Yolo5(model_path = MODEL_B_PATH)
        else:
            raise ValueError("Invalid mega version.")
        
        # load the classifier
        self.classifier = CNNS(model_name=class_name,model_path=class_path)

        # set the image size
        self.img_size = img_size

    def sgl_detect(self, image_path, classify = True, show=False, conf_thresh=0, format="std"):
        img = cv2.imread(image_path)  # Read the image
        results = self.yolo.model(img)  # Perform detection

        # extract the resulst and cut out the detections
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

                # set the detection save to true
                save = True

                if classify: # classify the detection if requested
                    cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                    crop_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"cropped.jpg")
                    cv2.imwrite(crop_save_path, cropped_img)
                    prediction = self.classifier.predict(crop_save_path)
                    os.remove(crop_save_path)
                    if prediction != 1: save = False

                if save:
                    detected_boxes.append({
                        'image_id': os.path.basename(image_path),
                        'bbox': box,  # Bounding box coordinates
                        'score': float(conf),  # Confidence score
                        'class': int(cls)  # Class label
                    })

        if show:
            self.yolo.draw_detection(detected_boxes, img, thresh=conf_thresh, format=format)

        return detected_boxes
    
    def test_detect(self, yolo_path, classify=True, conf_thresh=0):
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
            detections = self.sgl_detect(img_file, classify=classify, conf_thresh=conf_thresh,format="coco")  # Adjust conf_thresh as needed

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

    def inference_time(self, image_folder, classify = True):
        img_files = glob.glob(f"{image_folder}/*.jpg")
        inf_times = []

        for img_file in img_files:
            start_time = time.time()
            results = self.sgl_detect(img_file,classify=classify)
            end_time = time.time()
            inf_times.append(end_time - start_time)

        avg_inf = sum(inf_times) / len(inf_times)

        return avg_inf

    def evaluate(self, yolo_path, classify = True):
        pred_detections = self.test_detect(yolo_path=yolo_path,classify=classify)
        eval = EvaluateModel(yolo_path,pred_detections,self.img_size[0],self.img_size[1])
        results = eval.run_evaluation()
        results['inference'] = self.inference_time(os.path.join(yolo_path,"images","test"),classify=classify)
        return results

if __name__ == "__main__":
    mega = Mega(version = "b", class_path="ObjectDetection/Megadetector/resnet_test.pth", class_name="resnet50")
    # print(mega.sgl_detect(image_path="Data/Formated/yolo/images/test/At the meerkat burrow_47.jpg",show=True,classify=False,format="coco"))
    # jpg_files = glob.glob(os.path.join("Data/Formated/yolo/images/test", '*.jpg'))
    # print(mega.inference_time("Data/Formated/yolo/images/test",classify=True))
    # print(mega.evaluate(yolo_path="Data/Formated/yolo",classify=True))
    mega.yolo.detect_video("Data/YoutubeCameraTrap/At the meerkat burrow.mp4")
    # for file in jpg_files:
    #     print(mega.sgl_detect(image_path=file, show=True,classify=False))
    # coco_detections = mega.test_detect(yolo_path="Data/Formated/yolo",classify=False)
    # print(mega.yolo.evaluate(yolo_path="Data/Formated/yolo"))