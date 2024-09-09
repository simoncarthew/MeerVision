# CLASS SCRIPT PATHS
import sys
yolo5_path = 'ObjectDetection/Yolo5'
yolo8_path = 'ObjectDetection/Yolo8'
sys.path.append(yolo5_path)
sys.path.append(yolo8_path)

# IMPORTS
import pandas as pd
import os
from yolo5 import Yolo5
from yolo8 import Yolo8

# GLOBALS
TRAINING_CSV = 'ObjectDetection/Training/train.csv'
RESULTS_PATH = 'ObjectDetection/Training/results'
YOLO_DATA = 'Data/Formated/yolo'

# RESULTS_FOLDER
existing_results = [d for d in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, d))]
next_number = 0
while f'results{next_number}' in existing_results:
    next_number += 1
new_folder = f'results{next_number}'
RESULTS_PATH = os.path.join(RESULTS_PATH, new_folder)
os.makedirs(RESULTS_PATH)

# READ IN TRAINING CSV
train_df = pd.read_csv(TRAINING_CSV)

# SET STD TRAINING PARAMETERS
std_batch = 32
std_epochs = 30
std_lr = 0.01
std_augment = True
std_percval = 0.2
std_batchsize = 32
std_lr = 32
std_freeze = 0
std_new = False
std_augment = True
std_epochs = 30
std_percval = 0.2
std_obs_no = -1
std_md_z1_train_val = 1000
std_md_z2_trainval = 1000
std_md_test_no = 0


# LOOP OVER ALL TRAINING INSTANCES
for index, row in train_df.iterrows():

    # YOLOV5
    if "yolo5" in row["model"]:
        # get the model size
        size = row["model"][6:]

        # create yolo instance
        yolo = Yolo5(model_size = size)

        # set parameters

        # set save_path
        yolo.train(data_path=YOLO_DATA, epochs=1)

    # YOLOV8

    # MEGADETECTOR
    