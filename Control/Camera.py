from picamera2 import Picamera2
from time import sleep
import time
import glob
import shutil
import os
import json
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

        # make the sub save directory
        save_dir = os.path.join(save_dir,self.time_to_path(rtc.read_time(),ext=""))
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

        # save meta data
        meta = {"fps" : samp_period, "start_time" : start_time}
        with open(os.path.join(save_dir,"meta.json"), "w") as json_file:
            json.dump(meta, json_file, indent=4)

        while self.running:
            if self.total_duration is not None and self.run_time >= self.total_duration:
                break

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
            self.run_time = rtc.time_difference(start_time, current_time)["total_seconds"]

        self.running = False

    def get_average_size(self, rtc):
        # set the save_dir
        save_dir = os.path.join("Control", "Results", "size_images")
        
        # make the save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # create new thread
        self.capture_thread = threading.Thread(target=self._capture_images, args=(save_dir, 1, 20, rtc))
        self.running = True 
        self.capture_thread.start()

        sleep(25)

        jpg_files = glob.glob(os.path.join(save_dir, "*.jpg"))

        if not jpg_files:
            return 0  # Return 0 if no .jpg files are found

        # Get sizes of all .jpg files in bytes and convert to KB
        sizes_kb = [os.path.getsize(file) / 1024 for file in jpg_files]

        # Calculate average size
        average_size_kb = sum(sizes_kb) / len(sizes_kb)

        with open(os.path.join("Control", "Results", "avg_image_size.txt"), 'w') as file:
            file.write(f"Average size of .jpg files: {average_size_kb:.2f} KB")

        # remove the test images
        shutil.rm_tree(save_dir)
    
    def test_deployment_time(self, save_dir, rtc, fps=1):
        # make the save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # make the sub save directory
        save_dir = os.path.join(save_dir,self.time_to_path(rtc.read_time(),ext=""))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # calculte the campling period
        samp_period = 1.0 / fps

        # create new thread
        self.capture_thread = threading.Thread(target=self._capture_images, args=(save_dir, samp_period, None, rtc))
        self.running = True 
        self.capture_thread.start()

        # keep track of running time
        start_time = rtc.read_time()
        run_time = 0
        while self.is_capture_running():
            current_time = rtc.read_time()
            run_time = rtc.time_difference(start_time, current_time)["total_seconds"]
            with open(os.path.join("Control", "Results", f"total_run_{fps}.txt"), 'w') as file:
                file.write(f"Total seconds: {run_time} seconds")
            sleep(30)