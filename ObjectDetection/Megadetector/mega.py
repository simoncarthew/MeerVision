import cv2
import torch
import os
import sys

# import evaluation
eval_path = os.path.join("ObjectDetection")
sys.path.append(eval_path)
from Evaluate import EvaluateModel

# import yolo5
yolo_path = os.path.join("ObjectDetection","Yolo5")
sys.path.append(yolo_path)
from yolo5 import Yolo5

# model paths
MODEL_A_PATH = os.path.join("ObjectDetection","Megadetector","md_v5a.0.0.pt")
MODEL_B_PATH = os.path.join("ObjectDetection","Megadetector","md_v5b.0.0.pt")

class Mega:
    def __init__(self, img_size=(1280, 1280), version = "a"):
        if version == "a":
            self.yolo = Yolo5(model_path = MODEL_A_PATH)
        elif version == "b":
            self.yolo = Yolo5(model_path = MODEL_A_PATH)
        else:
            raise ValueError("Invalid mega version.")
        
        self.img_size = img_size

if __name__ == "__main__":
    mega = Mega(version = "a")
    # print(mega.detect(image_path="Data/Formated/yolo/images/test/Suricata_suricatta_1657.jpg"))