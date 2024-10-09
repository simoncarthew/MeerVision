# imports
import os
import pandas as pd
import argparse
import json
import sys
import glob
from ResultsSynthesis import filter_results, STD

# add relatove folders to system path
sys.path.append(os.path.join("ObjectDetection","Yolo"))

# import assistive classes
from yolo import Yolo

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
results_df = filter_results(results_df,['batch'])
print(results_df)

# create empty df if it doesnt exist
inferences_path = os.path.join(args.path,"inference_times.csv")
if os.path.exists(inferences_path):
    df = pd.read_csv(inferences_path)
    print("Loaded old results into data frame")
else:
    columns = ['model', 'pc', 'pi', 'pi_ncnn']
    df = pd.DataFrame(columns=columns)
    print("Created New Data frame.")

# iterate over yolo models
for idx, row in results_df.iterrows():
    model_name = row['model']
    model_path = os.path.join(args.path, "models", "model_" + str(row['id']) + ".pt")

    print(f"Loading {model_path}")
    model = Yolo(model_path=model_path, device='cpu')
    print(f"Loaded {model_name}")

    # get the inference time
    print("Starting inference test")
    avg_inf = model.inference_time(TEST_IMAGES)
    del model
    print("Inference test complete")

    pi_ncnn = 0
    if args.pi:
        model_path = os.path.join(args.path, "pi_models", "model_" + str(row['id']) + "_ncnn_model")
        model = Yolo(model_path=model_path, device='cpu')
        pi_ncnn = model.inference_time(TEST_IMAGES)

    # set the inference time for chosen device
    pc = 0
    pi = 0
    if args.pi: pi = avg_inf
    elif args.pc: pc = avg_inf

    # add results to inference time
    seen = False
    for idx, row in df.iterrows():
        if row['model'] == model_name:
            if device == 'pc': 
                pi = row['pi']
                pi_ncnn = row['pi_ncnn']
            if device == 'pi': pc = row['pc']
            df.loc[idx] = [model_name, pc, pi, pi_ncnn]
            seen = True
    if not seen:
        df.loc[len(df)] = [model_name, pc, pi, pi_ncnn]
    print("Results saved to df")
    df.to_csv(inferences_path, index = False)
    print("CSV saved")