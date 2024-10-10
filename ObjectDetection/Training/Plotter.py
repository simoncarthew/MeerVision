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

#### HYPER PARAMETER PLOTTING

def plot_hyper_param_impacts(in_df, metric, model, save_path, parameters=['batch', 'freeze', 'optimizer'], title = None):
    # create plot
    text_size = 15
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_ylim([0.0, 1])
    
    # initialize variables
    x_offset = 0
    bar_width = 0.3
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
        ax.text(group_center, -0.15, param.upper(), ha='center', va='top', fontsize=text_size + 3)
        
        x_offset += len(param_values) + 0.5
    
    # set the title and x labels
    if title: ax.set_title(title, fontsize=text_size + 5)
    ax.set_ylabel(metric.upper(), fontsize=text_size + 3)
    ax.set_xticks([])
    
    # horizontal line for std metric value
    ax.axhline(y=std_metric, color='r', linestyle='--', label='Standard', alpha=0.2)
    
    # set y-axis tick label size
    ax.tick_params(axis='y', labelsize=text_size)
    
    plt.title(f"Hyper Parameter Tuning YOLOv{model[0]} {metric.upper()}")
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
            plot_hyper_param_impacts(df, metric, model, os.path.join(model_res_path, f"hyp_tune_{metric}.jpg"))

#### MODEL SZ FUNCTIONS ####

def plot_model_size_accuracies(df, model_sizes = ['n', 's', 'm', 'l'], metrics = ['mAP50','f1'], save_path = os.path.join(PLOT_SZ_PATH,"model_size")):
    df = filter_results(df,['batch'])

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
        ax.set_ylabel(metric)
        ax.set_title(f'Model Size vs {metric} Score')
        ax.set_xticks(x + width / 2, model_sizes)
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 100)  # Assuming accuracy is between 0 and 1

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
        ax.set_ylim(0, 12000)  # Set y-axis limits

        # Save the figure
        plt.savefig(save_path + "_" + model + ".png")
        plt.close()

def plot_process_time(df, model_time = 'pi', deployment_time = 6, max_fps = 3, save_path = os.path.join(PLOT_SZ_PATH,"process_time")):
    models = ["8","5"]
    model_sizes = ["n", "s", "m", "l"]
    
    for model in models:
        filtered_df = filter_results(df,model=model)

        times = filtered_df[model_time].tolist()
        no_images = deployment_time * 60 * 60

        inference_times = {}
        for i in range(1,max_fps + 1):
            inference_times[f"{i} FPS"] = [round(num * no_images * i / 60 / 60,3) for num in times]

        x = np.arange(len(model_sizes))  # the label locations
        width = 0.3 
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for idx, (attribute, measurement) in enumerate(inference_times.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add labels, title, and customize x-axis tick labels
        ax.set_ylabel('Processing Time (Hours)')
        ax.set_xlabel('Model Sizes')
        ax.set_title(f'YOLOv{model} {deployment_time} Hour Deployment Process Time vs Model Sizes')

        # Correct x-tick positioning
        mid_bar_offset = width * (len(inference_times) - 1) / 2
        ax.set_xticks(x + mid_bar_offset)
        ax.set_xticklabels(model_sizes)
        ax.set_ylim(0, 210)  # Set y-axis limits
        ax.legend()

        # Save the figure
        plt.savefig(save_path + "_" + model + ".png")
        plt.close()

def plot_run_times(df, save_path=os.path.join(PLOT_SZ_PATH, "run_times.png"), bar_width=0.4, max_discharge=0.7, max_capacity=10):
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10,4))

    fps = df["fps"]
    y_pos = np.arange(len(fps))
    run_times = [round(max_capacity * max_discharge / amps, 2) for amps in df["amps"].tolist()]

    # Create a horizontal bar chart with adjustable bar width
    ax.barh(y_pos, run_times, height=bar_width, align='center', color=COLOURS)
    
    # Set y-ticks and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fps)
    ax.invert_yaxis()  # Invert the y-axis to have the highest FPS on top
    ax.set_xlabel('Run Time (hours)')
    ax.set_ylabel('FPS')
    ax.set_title('FPS vs Run Time')

    # Save the plot
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory

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
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    plt.plot(model_sizes, pi_speed[0], label='Yolo8', marker='o',color=COLOURS[0], linewidth=4)
    plt.plot(model_sizes, pi_speed[1], label='Yolo5', marker='o',color=COLOURS[1], linewidth=4)

    # Add labels and title
    plt.xlabel('Model Names')
    plt.ylabel('Speed-Up')
    plt.title(f'Pi vs PC Speed Up')
    plt.legend()

    plt.savefig(save_path + ".png")
    plt.close()

def plot_trends(df, save_path = os.path.join(PLOT_SZ_PATH,"trend")):
    no_epochs = 39

    for idx, row in df.iterrows():
        trends = pd.read_csv(os.path.join(MERGED_SZ_PATH,"trends","results_" + str(row['id']) + '.csv'))

        # Create a figure and two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # plot the validation acccuracy
        try:
            map_50 = trends['metrics/mAP50(B)'].tolist()[:no_epochs]
        except:
            map_50 = trends['metrics/mAP_0.5'].tolist()[:no_epochs]
        ax1.plot(trends['epoch'].tolist()[:no_epochs], map_50, color=COLOURS[0], label='val mAP50')
        ax1.set_ylabel('mAP50')
        ax1.set_xlabel('Epochs')
        ax1.legend()

        # Plot the second line graph on the bottom axis
        ax2.plot(trends['epoch'].tolist()[:no_epochs], trends['train/box_loss'].tolist()[:no_epochs], color=COLOURS[1], label='train loss')
        ax2.plot(trends['epoch'].tolist()[:no_epochs], trends['val/box_loss'].tolist()[:no_epochs], color=COLOURS[2], label='val loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Box Loss')
        ax2.legend()

        fig.suptitle(f"Training Trends for Model {row['model']}", fontsize=16)

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Show the plot
        plt.savefig(save_path + f"_{row['model']}.png")

def plot_obs_accuracy(df, save_path = os.path.join(PLOT_SZ_PATH,"obs_acc")):
    df = filter_results(df,['obs_no','md_z1_trainval', 'md_z2_trainval', 'optimizer'],model_size='n')
    df = df[(df['md_z1_trainval'] == 0) & (df['md_z2_trainval'] == 0)]
    
    obs_nos = df['obs_no'].unique().tolist()
    metrics = ['mAP50', 'f1']
    
    for metric in metrics:
        accuracies = {
            'Yolo5': list([round(num * 100, 2) for num in filter_results(df,model='5')[metric]]),
            'Yolo8': list([round(num * 100, 2) for num in filter_results(df,model='8')[metric]])
        }

        x = np.arange(len(obs_nos))
        width = 0.3
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for idx, (attribute, measurement) in enumerate(accuracies.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric)
        ax.set_xlabel("Number of Observed Images in Training Data")
        ax.set_title(f'Model Size vs {metric} Score')
        ax.set_xticks(x + width / 2, obs_nos)
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 100)  # Assuming accuracy is between 0 and 1

        plt.savefig(save_path + f"_{metric}.png")
        plt.close()

def plot_md_accuracy(df, save_path = os.path.join(PLOT_SZ_PATH,"md_acc")):
    df = filter_results(df,['md_z1_trainval', 'md_z2_trainval', 'optimizer'],model_size='n')
    df = df[(df['md_z1_trainval'] != 0) & (df['md_z2_trainval'] != 0)]
    
    md_nos = df['md_z1_trainval'].unique().tolist()
    md_nos.sort()
    metrics = ['mAP50', 'f1']
    
    for metric in metrics:
        models = ['Yolo5','Yolo8']
        accuracies = {'Yolo5':[],'Yolo8':[]}
        for model in models:
            filtered_df = filter_results(df,model=model[4])
            for md_no in md_nos:
                accuracies[model].append(round(filtered_df[(filtered_df['md_z1_trainval'] == md_no) & (filtered_df['md_z2_trainval'] == md_no)][metric].tolist()[0]*100,2))

        x = np.arange(len(md_nos))
        width = 0.3
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for idx, (attribute, measurement) in enumerate(accuracies.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric)
        ax.set_xlabel("Number of New Zealand Images in Training Data")
        ax.set_title(f'Model Size vs {metric} Score')
        ax.set_xticks(x + width / 2, md_nos)
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(0, 100) 

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
    exit()
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