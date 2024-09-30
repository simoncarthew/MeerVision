import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shutil
import json
import numpy as np

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
  'optimizer': 'SGD'}

def merge_results(directory, save_dir):
    # global results dataframe
    global_results = pd.DataFrame()
    global_model_id = 0

    # tends directory
    trends_dir = os.path.join(save_dir, 'trends')
    os.makedirs(trends_dir, exist_ok=True)

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

                    # extract trnds from results csv
                    if os.path.exists(results_csv_path):
                        trend_df = pd.read_csv(results_csv_path)
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

def filter_results(results_df, unfiltered_params = None, model = None, std=STD):

    # filter for desired model
    if model: results_df = results_df[(results_df['model'].str.contains(str(model), na=False))]

    # select filters
    if unfiltered_params:
        filters = {'batch':True, 'lr':True, 'augment':True, 'freeze':True, 'pretrained':True, 'obs_no':True, 'md_z1_trainval':True, 'md_z2_trainval':True, 'md_test_no':True, 'optimizer':True, 'img_sz':True}
        
        for unfiltered_param in unfiltered_params:
            filters[unfiltered_param] = False

        # filter for desired params
        for key, value in filters.items():
            if value: results_df = results_df[(results_df[key] == std[key])]

    return results_df

def plot_hyper_param_impacts(in_df, metric, save_path, parameters=['batch', 'lr', 'freeze', 'optimizer'], title = None):
    # create plot
    text_size = 15
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_ylim([0.0, 1])
    
    # initialize variables
    x_offset = 0
    bar_width = 0.15
    colors = matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, 10))
    
    for i, param in enumerate(parameters):
        # filter for desired variable
        df = filter_results(in_df, unfiltered_params=[param])

        # get the std metric value and other values
        std_value = STD[param]
        param_values = sorted(df[param].unique())
        std_metric = df[(df[param] == std_value)][metric].values[0]
        metrics = [df[(df[param] == val)][metric].values[0] for val in param_values]
        
        # create bar plots
        x = np.arange(len(param_values)) + x_offset
        rects = ax.bar(x, metrics, bar_width, label=param, color=colors[i])
        
        # add bar labels
        ax.bar_label(rects, padding=3, rotation=90, fontsize=text_size, fmt='%.3f')
        
        # add parameter values to x axis
        for j, val in enumerate(param_values):
            ax.text(x[j], -0.015, str(val), ha='center', va='top', rotation=90, fontsize=text_size)
        
        # add parameter name in uppercase
        group_center = x_offset + (len(param_values) - 1) / 2
        ax.text(group_center, -0.15, param.upper(), ha='center', va='top', fontsize=text_size + 3, fontweight='bold')
        
        x_offset += len(param_values) + 0.5
    
    # set the title and x labels
    if title: ax.set_title(title, fontsize=text_size + 5, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=text_size + 3, fontweight='bold')
    ax.set_xticks([])
    
    # horizontal line for std metric value
    ax.axhline(y=std_metric, color='r', linestyle='--', label='Standard', alpha=0.2)
    
    # set y-axis tick label size
    ax.tick_params(axis='y', labelsize=text_size)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_data_size_impacts(df, metric, save_path, dataset = 'Observed', model = None):
    # set the x and y axis
    y_col = metric
    if dataset == 'Observed': x_col = 'obs_no'
    elif dataset == 'MeerDown': x_col = 'md_z1_trainval'
    else:
        print('Invalid dataset chosen')
        exit()

    # filter the df
    if x_col == 'obs_no': df = filter_results(df, [x_col], model=model)
    else: df = filter_results(df, [x_col, 'md_z2_trainval'], model=model)

    # sort df
    df = df.sort_values(by=x_col, ascending=True)

    # define colour_scheme
    colors = matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, 10))
    
    # create plots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # bar positions and heights
    x = np.arange(len(df[x_col]))
    heights = df[y_col].values
    
    # plot the bars
    rects = ax.bar(x, heights, color=colors[:len(heights)])
    
    # set x ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].values, rotation=45, ha='right')
    
    # set labels and title
    ax.set_xlabel(dataset)
    ax.set_ylabel(y_col)
    ax.set_title(f'Bar Graph of {y_col} vs {x_col}')
    
    # save plot
    plt.savefig(save_path)
    plt.close()

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

# set the input directory
MERGED_PATH = os.path.join("ObjectDetection", "Training", "Results", "merged_results")
UNMERGED_PATH = os.path.join("ObjectDetection", "Training", "Results", "hyper_tune")

# set output directory and models
OUT_PATH = os.path.join("ObjectDetection", "Training", "Plots")
models = ["5s", "8n"]

# remove the folders and remerge results
if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)
    os.mkdir(OUT_PATH)
if os.path.exists(MERGED_PATH):
    shutil.rmtree(MERGED_PATH)
    os.mkdir(MERGED_PATH)
    merge_results(UNMERGED_PATH, MERGED_PATH)

for model in models:
    # make the output folder
    model_res_path = os.path.join(OUT_PATH,"yolo_" + model)
    os.mkdir(model_res_path)

    # read in merged results
    df = pd.read_csv(os.path.join(MERGED_PATH,"results.csv"))

    # filter for specific model
    df = filter_results(df,unfiltered_params=None, model=model)

    # set the desired metrics
    metrics = ['precision', 'recall', 'mAP50']

    # plot the hyper parameter results
    best_parameters = {}
    for metric in metrics:
        # plot hyper parameter impacts
        plot_hyper_param_impacts(df, metric, os.path.join(model_res_path, f"hyp_tune_{metric}.jpg"))

        # plot data size impacts
        plot_data_size_impacts(df, metric, os.path.join(model_res_path, f"meerdown_data_{metric}.jpg"), dataset='MeerDown', model = model)
        plot_data_size_impacts(df, metric, os.path.join(model_res_path, f"observed_data_{metric}.jpg"), dataset='Observed', model = model)

        # get best parameters
        parameters = get_best_parameters(df,metric)
        best_parameters[metric] = parameters
    
    # save best parameters
    with open(os.path.join(model_res_path, f"best_param.json"), "w") as f:
        json.dump(best_parameters, f, indent=4)

# print(filter_results(df,unfiltered_params=['obs_no'], model='5s'))