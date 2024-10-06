import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shutil
import json
import glob
import numpy as np

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
  'optimizer': 'SGD'}

COLOURS=matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, 10))

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
                    print(results_csv_path)

                    # move model to models folder
                    source = os.path.join(folder_path, "models", f"model_{model_id}", "weights", "best.pt")
                    destination = os.path.join(global_models_dir, f"model_{global_model_id}.pt")
                    shutil.copy(source, destination)

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

def filter_results(results_df, unfiltered_params = None, model = None, std=STD, filters = {'batch':True, 'lr':True, 'augment':True, 'freeze':True, 'pretrained':True, 'obs_no':True, 'md_z1_trainval':True, 'md_z2_trainval':True, 'md_test_no':True, 'optimizer':True, 'img_sz':True}):

    # filter for desired model
    if model: results_df = results_df[(results_df['model'].str.contains(str(model), na=False))]

    # select filters
    if unfiltered_params:
        
        for unfiltered_param in unfiltered_params:
            filters[unfiltered_param] = False

        # filter for desired params
        for key, value in filters.items():
            if value: results_df = results_df[(results_df[key] == std[key])]

    return results_df

##### HYPER PARAMATER FUNCTIONS ####

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

def plot_model_size(df):
    pass

##### PROCESS HYPER PARAMETER RESULTS ####

# set directories
UNMERGED_HYP_PATH = os.path.join("ObjectDetection", "Training", "Results", "hyper_tune")
MERGED_HYP_PATH = os.path.join("ObjectDetection", "Training", "Results", "merged_hyp_results")
PLOT_PATH = os.path.join(MERGED_HYP_PATH, "plots")
models = ["5s", "8n"]

# remove the folders 
if os.path.exists(MERGED_HYP_PATH):
    shutil.rmtree(MERGED_HYP_PATH)
os.mkdir(MERGED_HYP_PATH)
merge_results(UNMERGED_HYP_PATH, MERGED_HYP_PATH)

if os.path.exists(PLOT_PATH):
    shutil.rmtree(PLOT_PATH)
os.mkdir(PLOT_PATH)

for model in models:
    # make the output folder
    model_res_path = os.path.join(PLOT_PATH,"yolo_" + model)
    os.mkdir(model_res_path)

    # read in merged results
    df = pd.read_csv(os.path.join(MERGED_HYP_PATH,"results.csv"))

    # filter for specific model
    df = filter_results(df,unfiltered_params=None, model=model)

    # set the desired metrics
    metrics = ['mAP50', 'f1']

    # plot the hyper parameter results
    best_parameters = {}
    for metric in metrics:
        # plot hyper parameter impacts
        plot_hyper_param_impacts(df, metric, os.path.join(model_res_path, f"hyp_tune_{metric}.jpg"))

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

# set the directories
UNMERGED_SZ_PATH = os.path.join("ObjectDetection", "Training", "Results", "model_sizes")
MERGED_SZ_PATH = os.path.join("ObjectDetection", "Training", "Results", "merged_sz_results")
PLOT_SZ_PATH = os.path.join(MERGED_SZ_PATH, "plots")

def plot_model_size_accuracies(df, model_sizes = ['n', 's', 'm', 'l'], metric = 'mAP50', save_path = os.path.join(PLOT_SZ_PATH,"model_size_accuracies.png")):
    
    # initialize model accuracies
    yolo5_acc = []
    yolo8_acc = []

    for model_size in model_sizes:    
        filtered_df = df[(df['model'].str[5:].str.contains(str(model_size), na=False))]
        yolo5_score = filtered_df[(filtered_df['model'].str.contains(str('5'), na=False))][metric]
        yolo8_score = filtered_df[(filtered_df['model'].str.contains(str('8'), na=False))][metric]
        if len(yolo5_score) > 0: yolo5_acc.append(yolo5_score.iloc[0])
        else: yolo5_acc.append(0)
        if len(yolo8_score) > 0: yolo8_acc.append(yolo8_score.iloc[0])
        else: yolo8_acc.append(0)
    
    test_accuracies = {
        'Yolo5': list([round(num * 100, 2) for num in yolo5_acc]),
        'Yolo8': list([round(num * 100, 2) for num in yolo8_acc])
    }

    x = np.arange(len(model_sizes))
    width = 0.3
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for idx, (attribute, measurement) in enumerate(test_accuracies.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracies of Pretrained and Untrained Models')
    ax.set_xticks(x + width / 2, model_sizes)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 100)  # Assuming accuracy is between 0 and 1

    plt.savefig(save_path)

##### MODEL SIZE PROCESSING #####


# remove the folders 
if os.path.exists(MERGED_SZ_PATH):
    shutil.rmtree(MERGED_SZ_PATH)
os.mkdir(MERGED_SZ_PATH)

if os.path.exists(PLOT_SZ_PATH):
    shutil.rmtree(PLOT_SZ_PATH)
os.mkdir(PLOT_SZ_PATH)

# merge results
merge_results(UNMERGED_SZ_PATH,MERGED_SZ_PATH)

# plot model size vs accuracies
df = pd.read_csv(os.path.join(MERGED_SZ_PATH,"results.csv"))
plot_model_size_accuracies(df)