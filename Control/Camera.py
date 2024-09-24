from picamera2 import Picamera2
from time import sleep
import time
import os

class Camera:
    def __init__(self):
        self.camera = Picamera2()

    def time_to_path(self,time, ext = ".jpg"):
        return f"{time['year']}_{time['month']}_{time['day']}_{time['hours']}_{time['minutes']}_{time['seconds']}{ext}"

    def capture(self, save_path):
        # get the directory of saved image
        directory = os.path.dirname(save_path)

        # make the directory if it doesnt exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # capture the photo
        self.camera.start()
        sleep(0.2)
        self.camera.capture_file(save_path)
        self.camera.stop()