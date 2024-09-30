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

class Process:
    def __init__(self):
        self.yolo5 = Yolo5(model_size='s', device='cpu')