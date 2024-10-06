# imports
import os
import pandas as pd
import argparse
import json
import sys
import glob

# add relatove folders to system path
sys.path.append(os.path.join("ObjectDetection","Yolo"))
sys.path.append(os.path.join("ObjectDetection","Yolo5"))
sys.path.append(os.path.join("ObjectDetection","Yolo8"))

# import assistive classes
from yolo import Yolo
from yolo5 import Yolo5
from yolo8 import Yolo8

# set std folder paths
RESULTS = os.path.join("ObjectDetection", "Training", "Results", "merged_sz_results")
TEST_IMAGES = os.path.join("Data", "InferenceTesting")

# initialize the argpasers
parser = argparse.ArgumentParser(description="Set flags for pi and pc")

# add the arguments and get the arguments
parser.add_argument("--pi", action="store_true", help="Set pi to True")
parser.add_argument("--pc", action="store_true", help="Set pc to True")
parser.add_argument("--path", type=str, default=RESULTS, help="Specify the results path")
args = parser.parse_args()

print("Got arguments")

# set device
if args.pi: device = 'pi'
elif args.pc: device = 'pc'

# load the results csv 
results_df = pd.read_csv(os.path.join(args.path, "results.csv"))

print("Loaded Model paths")

# create empty df if it doesnt exist
inferences_path = os.path.join(args.path,"inference_times.csv")
if os.path.exists(inferences_path):
    df = pd.read_csv(inferences_path)
    print("Loaded old results into data frame")
else:
    columns = ['model', 'pc', 'pi']
    df = pd.DataFrame(columns=columns)
    print("Created New Data frame.")

# iterate over yolo models
for idx, row in results_df.iterrows():
    model_name = row['model']
    model_path = os.path.join(args.path, "models", "model_" + str(row['id']) + ".pt")
    print(model_path)

    # this is for the old yolo format
    if 'yolo5' in model_name:
        print("Loading yolo5")
        model = Yolo5(model_path=model_path, device='cpu')
        print("Loaded yolo5")
    else:
        print("Loading yolo8")
        model = Yolo8(model_path=model_path, device='cpu')
        print("Loaded yolo8")

    # get the inference time
    print("Starting inference test")
    avg_inf = model.inference_time(TEST_IMAGES)
    print("Inference test complete")

    # set the inference time for chosen device
    pc = None
    pi = None
    if args.pi: pi = avg_inf
    elif args.pc: pc = avg_inf

    # add results to inference time
    seen = False
    for idx, row in df.iterrows():
        if row['model'] == os.path.basename(model_path)[:-3]:
            if device == 'pc': pi = row['pi']
            if device == 'pi': pc = row['pc']
            df.loc[idx] = [os.path.basename(model_path)[:-3], pc, pi]
            seen = True
    if not seen:
        df.loc[len(df)] = [os.path.basename(model_path)[:-3], pc, pi]
    print("Results saved to df")
    df.to_csv(inferences_path, index = False)
    print("CSV saved")