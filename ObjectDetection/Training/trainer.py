# IMPORTS
import pandas as pd
import Yolo5

# READ IN TRAINING CSV
train_df = pd.read_csv('train.csv')

# SET STD TRAINING PARAMETERS
batch = 32
epochs = 30
lr = 0.01
augment = True
percval = 0.2
batchsize = 32
lr = 32
freeze = 0
new = False
augment = True
epochs = 30
percval = 0.2
obs_no = -1
md_z1_train_val = 1000
md_z2_trainval = 1000
md_test_no = 0


# LOOP OVER ALL TRAINING INSTANCES
for index, row in train_df.iterrows():
    # YOLOV5
    if "yolo5" in row["model"]:
        # get the model size
        size = row["model"][6:]

        # 

    # YOLOV8

    # MEGADETECTOR
    