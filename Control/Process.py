# standard imports
import sys
import os

# add relatove folders to system path
sys.path.append(os.path.join("ObjectDetection","Yolo5"))
sys.path.append(os.path.join("ObjectDetection","Yolo8"))
sys.path.append(os.path.join("ObjectDetection","Megadetector"))

# import assistive classes
from yolo5 import Yolo5
from yolo8 import Yolo8
from Mega import Mega

# model name to path maps
model_dir_path = os.path.join("Control","Models")

class Process:
    def __init__(self, model_name):
        if "5" in model_name:
            self.model = Yolo5(model_path = os.path.join(model_dir_path, model_name + ".pt"), device='cpu')
        elif "8" in model_name:
            self.model = Yolo8(model_path = os.path.join(model_dir_path, model_name + ".pt"), device='cpu')
        elif "megaA" in model_name:
            self.model = Mega(model_path = os.path.join(model_dir_path, model_name + ".pt"), device='cpu')