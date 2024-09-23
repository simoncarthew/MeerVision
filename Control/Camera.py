from picamera2 import Picamera2
from time import sleep
import os

class Camera:
    def __init__(self):
        # intialise the camera
        self.camera = Picamera2()
        
    def capture(self, save_path):
        # get the directory of saved image
        directory = os.path.dirname(save_path)

        # make the directory if it doesnt exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # capture the photo
        self.camera.start()
        sleep(0.01)
        self.camera.capture_file(save_path)
        self.camera.stop()

    def time_to_path(self,time, ext = ".jpg"):
        return f"{time["year"]}_{time["month"]}_{time["day"]}_{time['hours']}_{time['minutes']}_{time['seconds']}{ext}"
