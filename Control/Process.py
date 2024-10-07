# standard imports
import sys
import os
import pandas as pd

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

    def detect_all(self, deployment_path = "images"):
        pass

if __name__ == "__main__":
    process = Process()