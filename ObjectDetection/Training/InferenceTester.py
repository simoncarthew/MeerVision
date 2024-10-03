# imports
import os
import pandas as pd
import argparse
import json
import sys
import glob

# add relatove folders to system path
sys.path.append(os.path.join("ObjectDetection","Yolo5"))
sys.path.append(os.path.join("ObjectDetection","Yolo8"))
sys.path.append(os.path.join("ObjectDetection","Megadetector"))

# import assistive classes
from yolo5 import Yolo5
from yolo8 import Yolo8
from Mega import Mega

# set std folder paths
RESULTS = os.path.join("ObjectDetection", "Training", "Results", "merged_results")
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

# load the best models file
models_path = os.path.join(args.path, "model_sizes")
model_paths = sorted(glob.glob(os.path.join(models_path, '*.pt')))

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
for model_path in model_paths:

    # load the model
    if 'yolo5' in model_path:
        print("Loading yolo5")
        model = Yolo5(model_path=model_path, device='cpu')
        print("Loaded yolo5")
    elif 'yolo8' in model_path:
        print("Loading yolo8")
        model = Yolo8(model_path=model_path, device='cpu')
        print("Loaded yolo8")
    else:
        print(f"{model_path} is not a supported model")

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

    df.to_csv(inferences_path, index = False)