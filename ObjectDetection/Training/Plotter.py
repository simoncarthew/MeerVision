import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shutil
import json
import glob
import sys
import numpy as np
from ResultsSynthesis import filter_results, STD, UNMERGED_HYP_PATH, MERGED_HYP_PATH, PLOT_HYP_PATH, MERGED_SZ_PATH, UNMERGED_SZ_PATH

# set directories
PLOT_SZ_PATH = os.path.join(MERGED_SZ_PATH, "plots")
COLOURS=[x for i,x in enumerate(matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, 10))) if i!=1]

FIGSIZE = (8,6)
TITLESIZE = 15
AXISSIZE = 12
TICKSIZE = 10
BARWIDTH = 0.3

#### HYPER PARAMETER PLOTTING

def plot_hyper_param_impacts(in_df, metric, model, save_path, parameters=['batch', 'freeze', 'optimizer'], title = None):
    # create plot
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_ylim([0.0, 1])
    
    # initialize variables
    x_offset = 0
    bar_width = BARWIDTH
    colors = COLOURS
    
    for i, param in enumerate(parameters):
        # filter for desired variable
        df = filter_results(in_df, unfiltered_params=[param], model=model)

        # get the std metric value and other values
        std_value = STD[param]
        param_values = sorted(df[param].unique())
        std_metric = df[(df[param] == std_value)][metric].values[0]
        metrics = [df[(df[param] == val)][metric].values[0] for val in param_values]
        
        # create bar plots
        x = np.arange(len(param_values)) + x_offset
        rects = ax.bar(x, metrics, bar_width, label=param, color=colors[i])
        
        # add bar labels
        ax.bar_label(rects, padding=3, rotation=90, fontsize=TICKSIZE, fmt='%.3f')
        
        # add parameter values to x axis
        for j, val in enumerate(param_values):
            ax.text(x[j], -0.018, str(val), ha='center', va='top', rotation=90, fontsize=AXISSIZE)
        
        # add parameter name in uppercase
        param = param[0].upper() + param[1:]
        group_center = x_offset + (len(param_values) - 1) / 2
        ax.text(group_center, -0.18, param, ha='center', va='top', fontsize=AXISSIZE + 3)
        
        x_offset += len(param_values) + 0.5
    
    # set the title and x labels
    if title: ax.set_title(title, fontsize=AXISSIZE + 5)
    ax.set_ylabel(metric.upper(), fontsize=AXISSIZE + 3)
    ax.set_xticks([])
    
    # horizontal line for std metric value
    ax.axhline(y=std_metric, color='r', linestyle='--', label='Standard', alpha=0.2)
    
    # set y-axis tick label size
    ax.tick_params(axis='y', labelsize=AXISSIZE)
    
    plt.title(f"{metric.upper()} Hyper Parameter Scores YOLOv{model[0]}", fontsize=TITLESIZE)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#### PLOT HYPER PARAMETERS ####
def plot_hyp_param():
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
            plot_hyper_param_impacts(df, metric, model, os.path.join(model_res_path, f"hyp_tune_{model}_{metric}.jpg"))

#### MODEL SZ FUNCTIONS ####

def plot_model_size_accuracies(df, model_sizes = ['n', 's', 'm', 'l'], metrics = ['mAP50','precision', 'recall'], save_path = os.path.join(PLOT_SZ_PATH,"model_size")):
    df = df[(df["obs_no"] == -1) & (df["md_z1_trainval"] == 1000) & (df["md_z2_trainval"] == 1000)]

    for metric in metrics:
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
            'YOLOv5': list([round(num * 100, 2) for num in yolo5_acc]),
            'YOLOv8': list([round(num * 100, 2) for num in yolo8_acc])
        }

        x = np.arange(len(model_sizes))
        width = BARWIDTH
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained',figsize=FIGSIZE)

        for idx, (attribute, measurement) in enumerate(test_accuracies.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=3, fontsize=TICKSIZE)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric.upper(), fontsize=AXISSIZE)
        ax.set_xlabel("Model Size", fontsize=AXISSIZE)
        ax.set_title(f'{metric.upper()} Scores Across Model Sizes', fontsize=TITLESIZE)
        ax.set_xticks(x + width / 2, model_sizes)
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 100)

        plt.savefig(save_path + "_" + metric + ".png")
        plt.close()

def plot_inferences(df, save_path = os.path.join(PLOT_SZ_PATH,"inference_times")):
    models = ["8","5"]
    model_sizes = ["n", "s", "m", "l"]

    for model in models:
        filtered_df = filter_results(df,model=model)
        pc_times = []
        pi_times = []

        for model_size in model_sizes:    
            pc = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pc']
            pi = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pi']
            if len(pc) > 0: pc_times.append(pc.iloc[0])
            else: pc_times.append(0)
            if len(pi) > 0: pi_times.append(pi.iloc[0])
            else: pi_times.append(0)

        inference_times = {
            'PC': list([round(num * 1000, 0) for num in pc_times]),
            'Pi': list([round(num * 1000, 0) for num in pi_times]),
        }

        x = np.arange(len(model_sizes))  # the label locations
        width = BARWIDTH
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained', figsize=FIGSIZE)

        for idx, (attribute, measurement) in enumerate(inference_times.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=3, fontsize = TICKSIZE)
            multiplier += 1

        # Add labels, title, and customize x-axis tick labels
        ax.set_ylabel('Inference Time (ms)', fontsize = AXISSIZE)
        ax.set_xlabel('Model Size', fontsize = AXISSIZE)
        ax.set_title(f'Inference Times Across YOLOv{model} Model Sizes', fontsize = TITLESIZE)

        # Correct x-tick positioning
        mid_bar_offset = width * (len(inference_times) - 1) / 2
        ax.set_xticks(x + mid_bar_offset)
        ax.set_xticklabels(model_sizes)

        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 12000)

        # Save the figure
        plt.savefig(save_path + "_" + model + ".png")
        plt.close()

def plot_process_time(df, deployment_time = 6, max_fps = 3, save_path = os.path.join(PLOT_SZ_PATH,"process_time")):
    models = ["8","5"]
    model_sizes = ["n", "s", "m", "l"]
    model_times = ["pi", "pc"]

    for model_time in model_times:
        for model in models:
            filtered_df = filter_results(df,model=model)

            times = filtered_df[model_time].tolist()
            no_images = deployment_time * 60 * 60

            inference_times = {}
            for i in range(1,max_fps + 1):
                inference_times[f"{i} FPS"] = [round(num * no_images * i / 60 / 60,2) for num in times]

            x = np.arange(len(model_sizes))  # the label locations
            width = 0.3 
            multiplier = 0

            fig, ax = plt.subplots(layout='constrained', figsize=FIGSIZE)

            for idx, (attribute, measurement) in enumerate(inference_times.items()):
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
                ax.bar_label(rects, padding=3, fontsize=TICKSIZE)
                multiplier += 1

            # find max processing time
            max_proc = 0
            for key, value in inference_times.items():
                if max(value) > max_proc: max_proc = max(value)

            # Add labels, title, and customize x-axis tick labels
            ax.set_ylabel('Processing Time (Hours)', fontsize=AXISSIZE)
            ax.set_xlabel('Model Size', fontsize=AXISSIZE)
            ax.set_title(f'YOLOv{model} Processing Time for a {deployment_time} Hour Deployment', fontsize=TITLESIZE)

            # Correct x-tick positioning
            mid_bar_offset = width * (len(inference_times) - 1) / 2
            ax.set_xticks(x + mid_bar_offset)
            ax.set_xticklabels(model_sizes, fontsize=TICKSIZE)
            ax.set_ylim(0, max_proc + 10)  # Set y-axis limits
            ax.legend()

            # Save the figure
            plt.tight_layout()
            plt.savefig(save_path + "_" + model + "_" + model_time + ".png")
            plt.close()

def plot_run_times(df, save_path=os.path.join(PLOT_SZ_PATH, "run_times.png"), max_discharge=0.6, max_capacity=10):
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10,4))

    fps = df["fps"]
    run_times = [round(max_capacity * max_discharge / amps, 2) for amps in df["amps"].tolist()]

    # Create a line graph
    ax.plot(fps, run_times, marker='o', color=COLOURS[1], label="Deployment Time", linewidth=3)

    # Set axis labels and title
    ax.set_xlabel('FPS', fontsize=AXISSIZE)
    ax.set_ylabel('Deployment Time (Hours)', fontsize=AXISSIZE)
    ax.set_title('FPS vs Deployment Time', fontsize=TITLESIZE)

    # Add grid and customize ticks
    ax.tick_params(axis='both', which='major', labelsize=TICKSIZE)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_speed_up(df, save_path = os.path.join(PLOT_SZ_PATH,"speed_up")):
    models = ["8","5"]
    model_sizes = ["n", "s", "m", "l"]

    pi_speed = []
    for model in models:
        filtered_df = filter_results(df,model=model)
        speeds = []

        for model_size in model_sizes:    
            pc = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pc']
            pi = filtered_df[filtered_df['model'].str[5:].str.contains(str(model_size))]['pi']
            if len(pc) > 0: 
                speeds.append(pc.iloc[0] / pi.iloc[0])
            else: 
                speeds.append(0)
        pi_speed.append(speeds)

    # Create a line plot
    plt.figure(figsize=FIGSIZE)
    plt.plot(model_sizes, pi_speed[0], label='YOLOv8', marker='o',color=COLOURS[0], linewidth=3)
    plt.plot(model_sizes, pi_speed[1], label='YOLOv5', marker='o',color=COLOURS[1], linewidth=3)

    # Add labels and title
    plt.xlabel('Model Names', fontsize=AXISSIZE)
    plt.ylabel('Speed-Up', fontsize=AXISSIZE)
    plt.ylim([0,0.5])
    plt.title(f'Pi vs PC Speed-Up', fontsize=TITLESIZE)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.close()

def plot_trends(df, save_path = os.path.join(PLOT_SZ_PATH,"trend")):
    no_epochs = 30

    for idx, row in df.iterrows():
        trends = pd.read_csv(os.path.join(MERGED_SZ_PATH,"trends","results_" + str(row['id']) + '.csv'))

        # Create a figure and two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

        # plot the validation acccuracy
        try:
            map_50 = trends['metrics/mAP50(B)'].tolist()[:no_epochs]
        except:
            map_50 = trends['metrics/mAP_0.5'].tolist()[:no_epochs]
        ax1.plot(trends['epoch'].tolist()[:no_epochs], map_50, color=COLOURS[0], label='val mAP50', linewidth=3)
        ax1.set_ylabel('mAP50', fontsize=AXISSIZE)
        ax1.set_xlabel('Epochs', fontsize=AXISSIZE)
        ax1.legend()

        # Plot the second line graph on the bottom axis
        ax2.plot(trends['epoch'].tolist()[:no_epochs], trends['train/box_loss'].tolist()[:no_epochs], color=COLOURS[1], label='train loss', linewidth=3)
        ax2.plot(trends['epoch'].tolist()[:no_epochs], trends['val/box_loss'].tolist()[:no_epochs], color=COLOURS[2], label='val loss', linewidth=3)
        ax2.set_xlabel('Epochs', fontsize=AXISSIZE)
        ax2.set_ylabel('Box Loss', fontsize=AXISSIZE)
        ax2.legend()

        model_name = f"YOLOv{row['model'][4:6]}"
        fig.suptitle(f"Training Trends for {model_name}", fontsize=TITLESIZE)

        plt.tight_layout()
        plt.savefig(save_path + f"_{row['model']}.png")

def plot_obs_accuracy(df, save_path = os.path.join(PLOT_SZ_PATH,"obs_acc")):
    df = filter_results(df,['obs_no','md_z1_trainval', 'md_z2_trainval', 'optimizer'],model_size='n')
    df = df[(df['md_z1_trainval'] == 0) & (df['md_z2_trainval'] == 0)]
    
    obs_nos = df['obs_no'].unique().tolist()
    obs_nos[obs_nos.index(-1)] = 4500
    metrics = ['mAP50', 'precision', 'recall']
    
    for metric in metrics:
        accuracies = {
            'YOLOv5': list([round(num * 100, 2) for num in filter_results(df,model='5')[metric]]),
            'YOLOv8': list([round(num * 100, 2) for num in filter_results(df,model='8')[metric]])
        }

        x = np.arange(len(obs_nos))
        width = BARWIDTH
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained',figsize=FIGSIZE)

        for idx, (attribute, measurement) in enumerate(accuracies.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric.upper(), fontsize=AXISSIZE)
        ax.set_xlabel("Number of Observed Images in Training Data", fontsize=AXISSIZE)
        ax.set_title(f'{metric.upper()} Scores Across Observed Images in Training Data', fontsize=TITLESIZE)
        ax.set_xticks(x + width / 2, obs_nos, fontsize=TICKSIZE)
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 100)  # Assuming accuracy is between 0 and 1

        plt.tight_layout()
        plt.savefig(save_path + f"_{metric}.png")
        plt.close()

def plot_md_accuracy(df, save_path = os.path.join(PLOT_SZ_PATH,"md_acc")):
    df = filter_results(df,['md_z1_trainval', 'md_z2_trainval', 'optimizer'],model_size='n')
    df = df[(df['md_z1_trainval'] != 0) & (df['md_z2_trainval'] != 0)]
    
    md_nos = df['md_z1_trainval'].unique().tolist()
    md_nos.sort()
    metrics = ['mAP50', 'precision', 'recall']
    
    for metric in metrics:
        models = ['YOLOv5','YOLOv8']
        accuracies = {'YOLOv5':[],'YOLOv8':[]}
        for model in models:
            filtered_df = filter_results(df,model=model[5])
            for md_no in md_nos:
                accuracies[model].append(round(filtered_df[(filtered_df['md_z1_trainval'] == md_no) & (filtered_df['md_z2_trainval'] == md_no)][metric].tolist()[0]*100,2))

        x = np.arange(len(md_nos))
        width = BARWIDTH
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained',figsize=FIGSIZE)

        for idx, (attribute, measurement) in enumerate(accuracies.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=3, fontsize=TICKSIZE)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric.upper(), fontsize=AXISSIZE)
        ax.set_xlabel("Number of New Zealand Images in Training Data", fontsize=AXISSIZE)
        ax.set_title(f'{metric.upper()} Scores Across New Zealand Images in Training Data', fontsize=TITLESIZE)
        ax.set_xticks(x + width / 2, md_nos)
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 100) 

        plt.tight_layout()
        plt.savefig(save_path + f"_{metric}.png")
        plt.close()

##### MODEL SIZE PROCESSING #####
def plot_model_size():
    if os.path.exists(PLOT_SZ_PATH):
        shutil.rmtree(PLOT_SZ_PATH)
    os.mkdir(PLOT_SZ_PATH)

    # plot model size vs accuracies
    df = pd.read_csv(os.path.join(MERGED_SZ_PATH,"results.csv"))
    plot_obs_accuracy(df)
    plot_md_accuracy(df)
    plot_model_size_accuracies(df)
    plot_trends(df)

    # plot inference times and speed up
    df = pd.read_csv(os.path.join(MERGED_SZ_PATH,"inference_times.csv"))
    plot_inferences(df)
    plot_speed_up(df)
    plot_process_time(df)

    # plot run times
    df = pd.read_csv(os.path.join("Control/Results/power_consumption.csv"))
    plot_run_times(df)

if __name__== "__main__":
    plot_hyp_param()
    plot_model_size()