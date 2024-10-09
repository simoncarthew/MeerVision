import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shutil
import json
import glob
import sys
import subprocess
import numpy as np

# add relatove folders to system path
sys.path.append(os.path.join("ObjectDetection","Yolo"))

# import assistive classes
from yolo import Yolo

##### STD PARAMETERS #####

STD = {'batch': 32,
  'lr': 0.01,
  'augment': True,
  'freeze': 0,
  'pretrained': True,
  'obs_no': -1,
  'md_z1_trainval': 1000,
  'md_z2_trainval': 1000,
  'md_test_no': 0,
  'img_sz': 640,
  'optimizer': 'SGD'
  }

UNMERGED_HYP_PATH = os.path.join("ObjectDetection", "Training", "Results", "hyper_tune")
MERGED_HYP_PATH = os.path.join("ObjectDetection", "Training", "Results", "merged_hyp_results")
PLOT_HYP_PATH = os.path.join(MERGED_HYP_PATH, "plots")

UNMERGED_SZ_PATH = os.path.join("ObjectDetection", "Training", "Results", "grand")
MERGED_SZ_PATH = os.path.join("ObjectDetection", "Training", "Results", "merged_sz_results")

#### GENERIC FUNCTIONS #####

def merge_results(directory, save_dir):
    # global results dataframe
    global_results = pd.DataFrame()
    global_model_id = 0

    # tends and model directory
    trends_dir = os.path.join(save_dir, 'trends')
    os.makedirs(trends_dir, exist_ok=True)
    global_models_dir = os.path.join(save_dir, 'models')
    os.makedirs(global_models_dir, exist_ok=True)

    # iterate over present results directories
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path) and folder.startswith("results"):
            train_csv_path = os.path.join(folder_path, 'train.csv')
            
            if os.path.exists(train_csv_path):
                # read the train csv
                train_data = pd.read_csv(train_csv_path)
                train_data = train_data[~train_data.isna().any(axis=1)]

                # iterate over every models results
                for idx, row in train_data.iterrows():
                    # get model directory
                    model_id = row['id']
                    model_dir = os.path.join(folder_path,"models", f'model_{model_id}')
                    results_csv_path = os.path.join(model_dir, 'results.csv')

                    # move model to models folder
                    source = os.path.join(folder_path, "models", f"model_{model_id}", "weights", "best.pt")
                    destination = os.path.join(global_models_dir, f"model_{global_model_id}.pt")
                    shutil.copy(source, destination)

                    # extract trnds from results csv
                    if os.path.exists(results_csv_path):
                        trend_df = pd.read_csv(results_csv_path)
                        trend_df.columns = trend_df.columns.str.strip()
                        trend_df = trend_df.drop_duplicates()
                        trend_df.to_csv(os.path.join(trends_dir, f'results_{global_model_id}.csv'), index=False)

                    # update the row id and add to global df
                    row['id'] = global_model_id
                    global_results = pd.concat([global_results, pd.DataFrame([row])])

                    # increment global id
                    global_model_id += 1

    # save global results
    global_results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    print(f"Data merged successfully! {global_model_id} models processed.")

def filter_results(results_df, unfiltered_params = None, model = None, std=STD, model_size = None, filters = {'batch':True, 'lr':True, 'augment':True, 'freeze':True, 'pretrained':True, 'obs_no':True, 'md_z1_trainval':True, 'md_z2_trainval':True, 'md_test_no':True, 'optimizer':True, 'img_sz':True}):

    # filter for desired model
    if model: results_df = results_df[(results_df['model'].str.contains(str(model), na=False))]

    if model_size: results_df = results_df[(results_df['model'].str[4:].str.contains(str(model_size), na=False))]

    # select filters
    if unfiltered_params:
        
        for unfiltered_param in unfiltered_params:
            filters[unfiltered_param] = False

        # filter for desired params
        for key, value in filters.items():
            if value: results_df = results_df[(results_df[key] == std[key])]

    return results_df

##### HYPER PARAMATER FUNCTIONS ####

def get_best_parameters(df, metric):
    parameters = {'batch':None, 'lr':None, 'freeze':None, 'optimizer':None, 'obs_no':None}
    
    # get best std parameters
    for key, value in parameters.items():
        df_filt = filter_results(df,[key])
        parameters[key] = str(df_filt.loc[df_filt[metric].idxmax(),key])

    # get best data parameters
    df_filt = filter_results(df,['md_z1_trainval', 'md_z2_trainval'])
    parameters['md'] = str(df_filt.loc[df_filt[metric].idxmax(),'md_z1_trainval'])


    return parameters

def save_best_hyp_models(df, metric):
    # initialise best models
    model_names = ['yolo5s', 'yolo8n']
    best_models = {}
    for model_name in model_names:
        best_models[model_name] = {metric: 0, 'id': None}

    # iterate over the rows
    for idx, row in df.iterrows():
        if row[metric] > best_models[row['model']][metric]:
            best_models[row['model']][metric] = row[metric]
            best_models[row['model']]['id'] = row['id']
    
    # save them to the best_models_folder
    best_model_path = os.path.join(MERGED_HYP_PATH,"model_sizes")
    os.mkdir(best_model_path)
    for key, value in best_models.items():
        shutil.copy(
            os.path.join(MERGED_HYP_PATH, "models", f"model_{best_models[key]['id']}.pt"),
            os.path.join(best_model_path, f"{key}.pt")
        )

##### PROCESS HYPER PARAMETER RESULTS ####

def hyp_process():
    models = ["5s", "8n"]

    # remove the folders 
    if os.path.exists(MERGED_HYP_PATH):
        shutil.rmtree(MERGED_HYP_PATH)
    os.mkdir(MERGED_HYP_PATH)
    merge_results(UNMERGED_HYP_PATH, MERGED_HYP_PATH)

    if os.path.exists(PLOT_HYP_PATH):
        shutil.rmtree(PLOT_HYP_PATH)
    os.mkdir(PLOT_HYP_PATH)


    for model in models:

        # make the output folder
        model_res_path = os.path.join(PLOT_HYP_PATH,"yolo_" + model)
        os.mkdir(model_res_path)


        # read in merged results
        df = pd.read_csv(os.path.join(MERGED_HYP_PATH,"results.csv"))

        # filter for specific model
        df = filter_results(df,unfiltered_params=None, model=model)

        # set the desired metrics
        metrics = ['mAP50', 'f1']

        # get best parameters
        best_parameters = {}
        for metric in metrics:

            # get best parameters
            parameters = get_best_parameters(df,metric)
            best_parameters[metric] = parameters
        
        # save best parameters
        with open(os.path.join(model_res_path, f"best_param.json"), "w") as f:
            json.dump(best_parameters, f, indent=4)

    # save the best models
    df = pd.read_csv(os.path.join(MERGED_HYP_PATH,"results.csv"))
    best_models = save_best_hyp_models(df, 'f1')

##### MODEL SIZE FUNCTIONS #####

def make_pi(df, save_dir = os.path.join(MERGED_SZ_PATH,"pi_models")):
    os.mkdir(save_dir)

    # model directory
    model_dir = os.path.join(MERGED_SZ_PATH,"models")

    # filter df for non data manipulation
    filtered_df = filter_results(df,['batch'])
    model_paths = []
    for idx, row in filtered_df.iterrows():
        model_paths.append(os.path.join(model_dir,f"model_{row['id']}.pt"))

    for model_path in model_paths:
        yolo = Yolo(model_path=model_path)
        yolo.to_pi()
        os.remove(model_path[:-3] + ".torchscript")
        shutil.copytree(model_path[:-3] + "_ncnn_model", os.path.join(save_dir,os.path.basename(model_path)[:-3] + "_ncnn_model"))
        shutil.rmtree(model_path[:-3] + "_ncnn_model")

##### MODEL SIZE PROCESSING #####

def size_process():

    # remove the folders 
    if os.path.exists(MERGED_SZ_PATH):
        shutil.rmtree(MERGED_SZ_PATH)
    os.mkdir(MERGED_SZ_PATH)

    # merge results
    merge_results(UNMERGED_SZ_PATH,MERGED_SZ_PATH)

    # convert the models to pi format
    df = pd.read_csv(os.path.join(MERGED_SZ_PATH,"results.csv"))
    make_pi(df)

    # run inference tester
    subprocess.run(["python", "ObjectDetection/Training/InferenceTester.py", "--pc"])

if __name__ == "__main__":
    # hyp_process()
    size_process()