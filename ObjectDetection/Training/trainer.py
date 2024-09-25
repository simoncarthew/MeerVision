# IMPORTS
import pandas as pd
import os
import shutil
import logging
import yaml
import traceback
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Training Script with Config')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
args = parser.parse_args()

# READ CONFIG
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# GLOBALS PATHS
TRAINING_CSV = config['paths']['training']
RESULTS_PATH = config['paths']['results']
YOLO_PATH = config['paths']['yolo']
MEERDOWN_FRAMES = config['paths']['meerdown_frame']
MEERDOWN_ANNOT = config['paths']['meerdown_annot']
OBS_FRAMES = config['paths']['obs_frames']
OBS_ANNOT = config['paths']['obs_annot']
DATA_PATH = config['paths']['data_path']

# CLASS SCRIPT IMPORTS
import sys
yolo5_path = config['paths']['yolo5_path']
yolo8_path = config['paths']['yolo8_path']
data_path = 'Data'
sys.path.append(yolo5_path)
sys.path.append(yolo8_path)
sys.path.append(data_path)

from yolo5 import Yolo5
from yolo8 import Yolo8
from DataManager import DataManager

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

# Redirect print statements to log file as well
class PrintLogger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass  # This method is needed for Python 3 compatibility

sys.stdout = PrintLogger()

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info('Starting the training script.')

# READ IN TRAINING CSV
train_df = pd.read_csv(TRAINING_CSV)
train_df['mAP5095'] = None
train_df['mAP75'] = None
train_df['mAP50'] = None
train_df['AR5095'] = None
train_df['precision'] = None
train_df['recall'] = None
train_df['f1'] = None
train_df['inference'] = None
logging.info('Added testing columns to the training df')

# SET STD TRAINING PARAMETERS
BATCH = config['parameters']['batch']
LR = config['parameters']['lr']
AUGMENT = config['parameters']['augment']
PERCVAL = config['parameters']['percval']
FREEZE = config['parameters']['freeze']
PRETRAINED = config['parameters']['pretrained']
EPOCHS = config['parameters']['epochs']
OBS_NO = config['parameters']['obs_no']
MD_Z1_TRAINVAL = config['parameters']['md_z1_trainval']
MD_Z2_TRAINVAL = config['parameters']['md_z2_trainval']
MD_TEST_NO = config['parameters']['md_test_no']
IMG_SZ = config['parameters']['img_sz']
OPTIMIZER = config['parameters']['optimizer']
STD_PARAM = {
    "img_sz": IMG_SZ,"optimizer" : OPTIMIZER, "batch": BATCH, "epochs": EPOCHS, "lr": LR, "augment": AUGMENT, "percval": PERCVAL, "freeze": FREEZE, "pretrained": PRETRAINED, "obs_no": OBS_NO, "md_z1_trainval": MD_Z1_TRAINVAL, "md_z2_trainval": MD_Z2_TRAINVAL, "md_test_no": MD_TEST_NO
}
logging.info('Read in the STD parameters')

# ALTERING_STD
def check_std(row):
    var_dict = STD_PARAM.copy()
    new_data = False
    for key, value in var_dict.items():
        if row[key] != 'std':
            var_dict[key] = row[key]
            if key in ["img_sz","percval","obs_no","md_z1_trainval","md_z2_trainval","md_test_no"]: new_data = True
    return var_dict, new_data

def update_train_csv(df, row_index, parameters,results):
    # update from standard parameters
    for key, value in parameters.items():
        df.at[row_index, key] = value

    # add results
    df.at[row_index, 'mAP5095'] = results['mAP5095']
    df.at[row_index, 'mAP75'] = results['mAP75']
    df.at[row_index, 'mAP50'] = results['mAP50']
    df.at[row_index, 'AR5095'] = results['AR5095']
    df.at[row_index, 'inference'] = results['inference']
    df.at[row_index, 'precision'] = results['precision']
    df.at[row_index, 'recall'] = results['recall']
    df.at[row_index, 'f1'] = results['f1']

    return df

logging.info('Processing training instances.')

# LOOP OVER ALL TRAINING INSTANCES
for index, row in train_df.iterrows():
    logging.info(f'Starting model {row["model"]}.')

    # SET PARAMETERS
    parameters, new_data = check_std(row)
    logging.info('Set parameters.')

    # SET MODEL_PATH
    models_path = os.path.join(RESULTS_PATH,'models')

    # create new dataset if necessary
    yolo_path = os.path.join(YOLO_PATH,"dataset.yaml")
    yolo_dir_path = os.path.join(YOLO_PATH)
    if new_data:
        logging.info(f'Creating new yolo dataset for model_{row["id"]}')
        yolo_path = os.path.join(DATA_PATH,"yolo")
        yolo_dir_path = yolo_path
        dm = DataManager(parameters["percval"], MEERDOWN_ANNOT, MEERDOWN_FRAMES, OBS_ANNOT, OBS_FRAMES, debug = True)
        dm.create_yolo_dataset(int(parameters["obs_no"]),int(parameters["md_z1_trainval"]),int(parameters["md_z2_trainval"]),int(parameters["md_test_no"]),yolo_path)
        yolo_path = os.path.join(yolo_path,"dataset.yaml")
        logging.info(f'Created new YOLO dataset at {yolo_path}.')

    trained = True

    # YOLOV5
    if "yolo5" in row["model"]:
        try:
            # load and train
            size = row["model"][5:]
            yolo = Yolo5(model_size=size, pretrained=parameters["pretrained"])
            logging.info(f'Created {row["model"]} and starting training')
            if parameters["epochs"] > 0:
                yolo.train(
                    data_path=yolo_path,
                    lr=float(parameters["lr"]),
                    augment=parameters["augment"],
                    epochs=int(parameters["epochs"]),
                    batch_size=int(parameters["batch"]),
                    img_sz=parameters["img_sz"],
                    freeze=int(parameters["freeze"]),
                    optimizer=parameters["optimizer"],
                    save_path=models_path
                )
            logging.info(f'Trained {row["model"]}.')

        except Exception as e:
            logging.error(f'Error training {row["model"]}: {e}\n{traceback.format_exc()}')
            trained = False

        try:
            # evaluate model
            logging.info(f'Evaluating model {row["model"]}')
            best_path = os.path.join(models_path, 'train','weights','best.pt')
            yolo = Yolo5(model_path = best_path)
            results = yolo.evaluate(yolo_dir_path,parameters["img_sz"],parameters["img_sz"])

        except Exception as e:
            logging.error(f'Error evaluating {row["model"]}: {e}\n{traceback.format_exc()}')
            trained = False

    # YOLOV8
    elif "yolo8" in row["model"]:
        try:

            # load and train
            size = row["model"][5:]
            yolo = Yolo8(model_size=size, pretrained=parameters["pretrained"])
            logging.info(f'Created {row["model"]}({row["id"]}) and starting training')
            if parameters["epochs"] > 0:
                yolo.train(
                    dataset_path=yolo_path,
                    lr=float(parameters["lr"]),
                    augment=bool(parameters["augment"]),
                    epochs=int(parameters["epochs"]),
                    batch=int(parameters["batch"]),
                    img_sz=int(parameters["img_sz"]),
                    freeze=int(parameters["freeze"]),
                    optimizer=parameters["optimizer"],
                    save_path=models_path
                )
            logging.info(f'Trained {row["model"]}({row["id"]}).')
        except Exception as e:
            logging.error(f'Error training {row["model"]}: {e}\n{traceback.format_exc()}')
            trained = False

        try:
            # evaluate model
            logging.info(f'Evaluating model {row["model"]}({row["id"]})')
            best_path = os.path.join(models_path, 'train','weights','best.pt')
            yolo = Yolo8(model_path = best_path)
            results = yolo.evaluate(yolo_dir_path,parameters["img_sz"],parameters["img_sz"])

        except Exception as e:
            logging.error(f'Error evaluating {row["model"]}: {e}\n{traceback.format_exc()}')
            trained = False

    # Update the train_df with the actual parameters
    if trained:
        train_df = update_train_csv(train_df, index, parameters, results)
        updated_train_csv_path = os.path.join(RESULTS_PATH, 'train.csv')
        train_df.to_csv(updated_train_csv_path, index=False)
        logging.info('Updated the training results')

        # rename train directory
        train_dir = os.path.join(models_path, 'train')
        if os.path.exists(train_dir):
            os.rename(train_dir, os.path.join(models_path, 'model_' + str(row["id"])))
            logging.info(f'Renamed results directory to model_{row["id"]}.')
        else:
            logging.error(f'Train directory not found: {train_dir}')
    
    if new_data:
        shutil.rmtree(os.path.join(DATA_PATH,"yolo"))

    logging.info(f'Completed model {row["model"]}({row["id"]})')

# SAVE UPDATED TRAINING_DF
updated_train_csv_path = os.path.join(RESULTS_PATH, 'train.csv')
train_df.to_csv(updated_train_csv_path, index=False)
logging.info(f'Saved final training CSV to {updated_train_csv_path}.')