from picamera2 import Picamera2
from time import sleep
import time
import os
import threading

class Camera:
    def __init__(self):
        self.camera = Picamera2()
        self.running = False

    def time_to_path(self,time, index = None, ext = ".jpg"):
        if index:
            path = f"{time['year']}_{time['month']}_{time['day']}_{time['hours']}_{time['minutes']}_{time['seconds']}_{index}{ext}"
        else:
            path = f"{time['year']}_{time['month']}_{time['day']}_{time['hours']}_{time['minutes']}_{time['seconds']}{ext}"
        return path

    def capture(self, save_path, setup_time = 0.2):
        # get the directory of saved image
        directory = os.path.dirname(save_path)

        # make the directory if it doesnt exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # capture the photo
        self.camera.start()
        sleep(setup_time)
        self.camera.capture_file(save_path)
        self.camera.stop()

    def start_capture_period(self, save_dir, rtc, fps=1, duration=10):
        # make the save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # calculte the campling period
        samp_period = 1.0 / fps

        # create new thread
        self.capture_thread = threading.Thread(target=self._capture_images, args=(save_dir, samp_period, duration, rtc))
        self.running = True 
        self.capture_thread.start()

    def stop_capture_period(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join() 

    def is_capture_running(self):
        if self.capture_thread:
            return self.capture_thread.is_alive()
        return False

    def _capture_images(self, save_dir, samp_period, duration, rtc):
        self.total_duration = duration
        self.running = True
        start_time = rtc.read_time()
        self.run_time = 0
        prev_time = start_time
        index = 0
        while self.running and self.run_time < self.total_duration:
            # get current time
            current_time = rtc.read_time()

            # check if its a new time stamp
            if current_time == prev_time and self.run_time != 0:
                index += 1
            else:
                index = 0

            # generate teh file path
            save_path = os.path.join(save_dir, self.time_to_path(current_time,index=index))
            
            # capture the image
            setup_time = 0.2
            self.capture(save_path, setup_time=setup_time)

            # wait for next sample
            sleep(samp_period - setup_time)

            prev_time = current_time
            self.run_time = rtc.time_difference(start_time, current_time)["seconds"]

        self.running = False