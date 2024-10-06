
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shutil
import json
import glob
import sys
import numpy as np
from ResultsSynthesis import filter_results, STD, UNMERGED_HYP_PATH, MERGED_HYP_PATH, PLOT_HYP_PATH, MERGED_SZ_PATH, UNMERGED_SZ_PATH, COLOURS

# set directories
PLOT_SZ_PATH = os.path.join(MERGED_SZ_PATH, "plots")

#### HYPER PARAMETER PLOTTING

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


#### PLOT HYPER PARAMETERS ####

# read in merged results
df = pd.read_csv(os.path.join(MERGED_HYP_PATH,"results.csv"))

# set the desired metrics
metrics = ['mAP50', 'f1']
models = ["5s", "8n"]

for model in models:

    # make the output folder
    model_res_path = os.path.join(PLOT_HYP_PATH,"yolo_" + model)

    # plot the hyper parameter results
    for metric in metrics:

        # plot hyper parameter impacts
        plot_hyper_param_impacts(df, metric, os.path.join(model_res_path, f"hyp_tune_{metric}.jpg"))


#### MODEL SZ FUNCTIONS ####

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
    ax.set_title('Yolo 5 and 8 Test Accuracies Across Model Sizes')
    ax.set_xticks(x + width / 2, model_sizes)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 100)  # Assuming accuracy is between 0 and 1

    plt.savefig(save_path)

def plot_inferences(df, save_path = os.path.join(PLOT_SZ_PATH,"inference_times")):
    models = ["8","5"]
    model_sizes = ["n", "s", "m", "l"]

    for model in models:
        filtered_df = filter_results(df,model=model)
        pc_times = []
        pi_times = []
        pi_ncnn_times = []

        for model_size in model_sizes:    
            pc = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pc']
            pi = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pi']
            pi_ncnn = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pi_ncnn']
            if len(pc) > 0: pc_times.append(pc.iloc[0])
            else: pc_times.append(0)
            if len(pi) > 0: pi_times.append(pi.iloc[0])
            else: pi_times.append(0)
            if len(pi_ncnn) > 0: pi_ncnn_times.append(pi_ncnn.iloc[0])
            else: pi_ncnn_times.append(0)

        inference_times = {
            'PC': list([round(num * 1000, 0) for num in pc_times]),
            'Pi': list([round(num * 1000, 0) for num in pi_times]),
            'PiNCNN': list([round(num * 1000, 0) for num in pi_ncnn_times])
        }

        x = np.arange(len(model_sizes))  # the label locations
        width = 0.3  # width of each bar
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for idx, (attribute, measurement) in enumerate(inference_times.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add labels, title, and customize x-axis tick labels
        ax.set_ylabel('Inference Time (ms)')
        ax.set_xlabel('Model Sizes')
        ax.set_title(f'YOLOv{model} Inference Times vs Model Sizes')

        # Correct x-tick positioning
        mid_bar_offset = width * (len(inference_times) - 1) / 2
        ax.set_xticks(x + mid_bar_offset)
        ax.set_xticklabels(model_sizes)

        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 1000)  # Set y-axis limits

        # Save the figure
        plt.savefig(save_path + "_" + model + ".png")

def plot_speed_up(df, save_path = os.path.join(PLOT_SZ_PATH,"speed_up")):
    models = ["8","5"]
    model_sizes = ["n", "s", "m", "l"]

    for model in models:
        filtered_df = filter_results(df,model=model)
        pi_speed = []
        pi_ncnn_speed = []

        for model_size in model_sizes:    
            pc = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pc']
            pi = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pi']
            pi_ncnn = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pi_ncnn']
            if len(pc) > 0: 
                pi_speed.append(pi.iloc[0] / pc.iloc[0])
                pi_ncnn_speed.append(pi_ncnn.iloc[0] / pc.iloc[0])
            else: 
                pi_speed.append(0)
                pi_ncnn_speed.append(0)

        # Create a line plot
        plt.figure(figsize=(10, 6))  # Optional: Set the figure size
        plt.plot(model_sizes, pi_speed, label='Pi', marker='o',color=COLOURS[0], linewidth=4)
        plt.plot(model_sizes, pi_ncnn_speed, label='PiNCNN', marker='o', color=COLOURS[1], linewidth=4)

        # Add labels and title
        plt.xlabel('Model Names')
        plt.ylabel('Speed-Up')
        plt.title(f'Yolo {model} Pi and PiNCNN vs PC Speed Up')
        plt.legend()

        plt.savefig(save_path + "_" + model + ".png")

##### MODEL SIZE PROCESSING #####

if os.path.exists(PLOT_SZ_PATH):
    shutil.rmtree(PLOT_SZ_PATH)
os.mkdir(PLOT_SZ_PATH)

# plot model size vs accuracies
df = pd.read_csv(os.path.join(MERGED_SZ_PATH,"results.csv"))
plot_model_size_accuracies(df)

# plot inference times and speed up
df = pd.read_csv(os.path.join(MERGED_SZ_PATH,"inference_times.csv"))
plot_inferences(df)
plot_speed_up(df)