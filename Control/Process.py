# standard imports
import sys
import os
import glob
import pandas as pd
import json
from natsort import natsorted
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import shutil

# add relatove folders to system path
sys.path.append(os.path.join("ObjectDetection","Yolo"))

# import assistive classes
from yolo import Yolo

# model name to path maps
RESULTS_DIR_PATH = os.path.join("ObjectDetection", "Training", "Results", "merged_sz_results")
MODEL_DIR_PATH = os.path.join(RESULTS_DIR_PATH, "models")

def paths_to_models():
    models = {}
    df = pd.read_csv(os.path.join(RESULTS_DIR_PATH,"results.csv"))
    for idx, row in df.iterrows():
        models[row['model']] = os.path.join(MODEL_DIR_PATH, f"model_{row['id']}.pt")
    return models

class Process:
    def __init__(self, version = "5", size = "nu"):
        # convert model version to model path
        models = paths_to_models()
        print("Got model paths.")

        # load selected model
        self.model = Yolo(model_path = models[f"yolo{version}{size}"])
        print(f"Loaded yolo{version}{size}")

    def detect_all(self, deployment_path, rendered_box_path=None, conf_thresh=0.0):
        # get all images
        images = glob.glob(os.path.join(os.path.join(deployment_path), "*.jpg"))
        images = natsorted(images)

        # detect all meerkats
        global_results = []
        for image_path in images:
            results = self.model.sgl_detect(image_path, show=False, conf_thresh=conf_thresh, format="yolo",save_path=os.path.join(rendered_box_path,os.path.basename(image_path)))
            global_results.append(results)

        with open(os.path.join(deployment_path,"meta.json"), 'r') as file:
            data = json.load(file)

        return global_results, [os.path.basename(image) for image in images], data.get("fps"), data.get("start_time")
    
    def synthesize_results(self,results, fps, start_time, images, save_path):
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

        # create df and store as csv
        df = pd.DataFrame(columns=["time", "image", "no_detections"])
        for i in range(len(time_intervals)):
            df.loc[len(df)] = [time_intervals[i], images[i],no_detections[i]]
        df.to_csv(save_path,index=False)

        return df

    def plot_results(self,df , save_path):
        plt.figure(figsize=(10, 5))
        plt.xlabel('TIME')
        plt.ylabel('NUMBER OF MEERKATS')
        plt.title('MEERKATS OVER TIME')
        plt.xticks(rotation=45)
        plt.ylim((0,max(df["no_detections"].tolist()) + 1))
        plt.plot(df["time"], df["no_detections"], marker='o', linestyle='-')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(save_path)

if __name__ == "__main__":
    process = Process(version="5", size="mu")
    deployment_path = "Control/Images/Deployments/2024_10_7_19_27_53"
    process_path = "Control/Processed/2024_10_7_19_27_53"
    if os.path.exists(process_path): 
        shutil.rmtree(process_path)
    os.mkdir(process_path)
    os.mkdir(os.path.join(process_path,"detections"))
    global_results, images, fps, start_time = process.detect_all(deployment_path=deployment_path,rendered_box_path=os.path.join(process_path,"detections"))
    df = process.synthesize_results(global_results, fps, start_time, images, save_path=os.path.join(process_path,"meerkats_vs_time.csv"))
    process.plot_results(df,save_path=os.path.join(process_path,"meerkats_vs_time.png"))