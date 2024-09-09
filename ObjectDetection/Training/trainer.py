# CLASS SCRIPT PATHS
import sys
yolo5_path = 'ObjectDetection/Yolo5'
yolo8_path = 'ObjectDetection/Yolo8'
data_path = 'Data'
sys.path.append(yolo5_path)
sys.path.append(yolo8_path)
sys.path.append(data_path)

# IMPORTS
import pandas as pd
import os
import shutil
import logging
from yolo5 import Yolo5
from yolo8 import Yolo8
from DataManager import DataManager

# GLOBALS PATHS
TRAINING_CSV = 'ObjectDetection/Training/train.csv'
RESULTS_PATH = 'ObjectDetection/Training/Results'
YOLO_PATH = 'Data/Formated/yolo'
MEERDOWN_FRAMES = 'Data/MeerDown/frames'
MEERDOWN_ANNOT = 'Data/MeerDown/raw/annotations.json'
OBS_FRAMES = 'Data/Observed/frames'
OBS_ANNOT = 'Data/Observed/annotations.json'
DATA_PATH = 'ObjectDetection/Training/Data'

# RESULTS_FOLDER
existing_results = [d for d in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, d))]
next_number = 0
while f'results{next_number}' in existing_results:
    next_number += 1
new_folder = f'results{next_number}'
RESULTS_PATH = os.path.join(RESULTS_PATH, new_folder)
os.makedirs(RESULTS_PATH)

# SET UP LOGGING
log_file = os.path.join(RESULTS_PATH, 'training_log.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info('Starting the training script.')

# SAVE CURRENT TRAIN FILE
shutil.copy(TRAINING_CSV, os.path.join(RESULTS_PATH,'train.csv'))
logging.info(f'Saved training CSV to {RESULTS_PATH}.')

# READ IN TRAINING CSV
train_df = pd.read_csv(TRAINING_CSV)

# SET STD TRAINING PARAMETERS
BATCH = 32
LR = 0.01
AUGMENT = True
PERCVAL = 0.2
FREEZE = 0
PRETRAINED = True
EPOCHS = 1
OBS_NO = -1
MD_Z1_TRAINVAL = 1000
MD_Z2_TRAINVAL = 1000
MD_TEST_NO = 0
IMG_SZ=640
OPTIMIZER = 'SGD'
STD_PARAM = {
    "img_sz": IMG_SZ,"optimizer" : OPTIMIZER, "batch": BATCH, "epochs": EPOCHS, "lr": LR, "augment": AUGMENT, "percval": PERCVAL, "freeze": FREEZE, "pretrained": PRETRAINED, "obs_no": OBS_NO, "md_z1_trainval": MD_Z1_TRAINVAL, "md_z2_trainval": MD_Z2_TRAINVAL, "md_test_no": MD_TEST_NO
}

# ALTERING_STD
def check_std(row):
    var_dict = STD_PARAM.copy()
    new_data = False
    for key, value in var_dict.items():
        if row[key] != 'std':
            var_dict[key] = row[key]
            if key in ["img_sz","percval","obs_no","md_z1_trainval","md_z2_trainval","md_test_no"]: new_data = True
    return var_dict, new_data

logging.info('Processing training instances.')

# LOOP OVER ALL TRAINING INSTANCES
for index, row in train_df.iterrows():
    logging.info(f'Processing row {index} with model {row["model"]}.')

    # SET PARAMETERS
    parameters, new_data = check_std(row)

    # create new dataset if necessary
    yolo_path = os.path.join(YOLO_PATH,"dataset.yaml")
    if new_data:
        yolo_path = os.path.join(DATA_PATH,"yolo")
        dm = DataManager(parameters["percval"], MEERDOWN_ANNOT, MEERDOWN_FRAMES, OBS_ANNOT, OBS_FRAMES, debug = False)
        dm.create_yolo_dataset(parameters["obs_no"],parameters["md_z1_trainval"],parameters["md_z2_trainval"],parameters["md_test_no"],yolo_path)
        yolo_path = os.path.join(yolo_path,"dataset.yaml")
        logging.info(f'Created new YOLO dataset at {yolo_path}.')

    # YOLOV5
    if "yolo5" in row["model"]:
        try:
            # load and train
            size = row["model"][5:]
            yolo = Yolo5(model_size=size, pretrained=parameters["pretrained"])
            yolo.train(
                data_path=yolo_path,
                lr=parameters["lr"],
                augment=parameters["augment"],
                epochs=parameters["epochs"],
                batch_size=parameters["batch"],
                img_sz=parameters["img_sz"],
                freeze=parameters["freeze"],
                optimizer=parameters["optimizer"]
            )
            logging.info(f'Trained YOLOv5 model with size {size}.')

            # change the name of the results
            os.rename(os.path.join(RESULTS_PATH, 'train'), os.path.join(RESULTS_PATH, 'model_' + str(row["id"])))
            logging.info(f'Renamed results directory to model_{row["id"]}.')
        except Exception as e:
            logging.error(f'Error training {row["model"]}: {e}')

    # YOLOV8
    if "yolo8" in row["model"]:
        try:
            # load and train
            size = row["model"][5:]
            yolo = Yolo8(model_size=size, pretrained=parameters["pretrained"])
            yolo.train(
                dataset_path=yolo_path,
                lr=parameters["lr"],
                augment=parameters["augment"],
                epochs=parameters["epochs"],
                batch=parameters["batch"],
                img_sz=parameters["img_sz"],
                freeze=parameters["freeze"],
                optimizer=parameters["optimizer"]
            )
            logging.info(f'Trained YOLOv8 model with size {size}.')

            # change the name of the results
            os.rename(os.path.join(RESULTS_PATH, 'train'), os.path.join(RESULTS_PATH, 'model_' + str(row["id"])))
            logging.info(f'Renamed results directory to model_{row["id"]}.')
        except Exception as e:
            logging.error(f'Error training {row["model"]}: {e}')