# standard imports
import sys
import os
import glob
import pandas as pd
import json
from natsort import natsorted
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# add relatove folders to system path
sys.path.append(os.path.join("ObjectDetection","Yolo"))

# import assistive classes
from yolo import Yolo

# model name to path maps
RESULTS_DIR_PATH = os.path.join("ObjectDetection", "Training", "Results", "merged_sz_results")
MODEL_DIR_PATH = os.path.join(RESULTS_DIR_PATH, "pi_models")

def paths_to_models():
    models = {}
    df = pd.read_csv(os.path.join(RESULTS_DIR_PATH,"results.csv"))
    for idx, row in df.iterrows():
        models[row['model']] = os.path.join(MODEL_DIR_PATH, f"model_{row['id']}_ncnn_model")
    return models

class Process:
    def __init__(self, version = "5", size = "s"):
        # convert model version to model path
        models = paths_to_models()
        print("Got model paths.")

        # load selected model
        self.model = Yolo(model_path = models[f"yolo{version}{size}"])
        print(f"Loaded yolo{version}{size}")

    def detect_all(self, deployment_path = "Control/Images/Deployments/2024_10_7_19_27_53"):
        # get all images
        images = glob.glob(os.path.join(os.path.join(deployment_path), "*.jpg"))
        images = natsorted(images)

        # detect all meerkats
        global_results = []
        for image_path in images:
            results = self.model.sgl_detect(image_path, show=False, conf_thresh=0, format="yolo")
            global_results.append()

        with open(os.path.join(deployment_path,"meta.json"), 'r') as file:
            data = json.load(file)

        return global_results, data.get("fps"), data.get("start_time")
    
    def synthesize_results():
        pass


    def plot_results(self,results, fps, start_time, save_path = os.path.join("number_of_detections")):

        # cont number of detections
        no_detections = [len(image) for image in results]

        # convert start time to datetime object
        start_time = datetime(
            year=start_time["year"],
            month=start_time["month"],
            day=start_time["day"],
            hour=start_time["hours"],
            minute=start_time["minutes"],
            second=start_time["seconds"]
        )

        # generate time intervals
        time_intervals = [start_time + timedelta(seconds=i / fps) for i in range(len(no_detections))]

        # Plotting the graph
        plt.figure(figsize=(10, 5))
        plt.plot(time_intervals, no_detections, marker='o', linestyle='-', color='b')
        plt.xlabel('Time')
        plt.ylabel('Number of Apples')
        plt.title('Apples Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        

if __name__ == "__main__":
    process = Process()
    process.detect_all()