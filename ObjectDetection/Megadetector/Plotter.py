
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shutil
import json
import glob
import sys
import numpy as np

# set directories
RES_PATH = os.path.join("ObjectDetection", "Megadetector", "Results")
PLOT_PATH = os.path.join(RES_PATH, "plots")
COLOURS=[x for i,x in enumerate(matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, 10))) if i!=1]

FIGSIZE = (8,6)
TITLESIZE = 15
AXISSIZE = 12
TICKSIZE = 10
BARWIDTH = 0.3

def plot_accuracies(df, save_path = os.path.join(PLOT_PATH,"mega")):
    metrics = ["mAP50", "precision", "recall", "inference"]

    for metric in metrics:
        versions = df["version"].unique()

        accs = {True:[],False:[]}
        for version in versions:
            for classifier in [True,False]:
                filtered_df = df[(df['version'] == version) & (df['classify'] == classifier)]
                accs[classifier].append(filtered_df.iloc[0][metric])
        
        if metric != "inference":
            test_accuracies = {
                    'Classifier': list([round(num, 3) for num in accs[True]]),
                    'NoClassifier': list([round(num, 3) for num in accs[False]])
                }
        else:
            test_accuracies = {
                    'Classifier': list([round(num * 1000, 0) for num in accs[True]]),
                    'NoClassifier': list([round(num * 1000, 0) for num in accs[False]])
                }

        x = np.arange(len(versions))
        width = 0.4
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained',figsize=FIGSIZE)

        for idx, (attribute, measurement) in enumerate(test_accuracies.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=2, fontsize=TICKSIZE)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xticks(x + width / 2, versions)
        ax.legend(loc='upper left', ncols=2)
        ax.set_xlabel("MegaDetector Version", fontsize=AXISSIZE)
        if metric != "inference": 
            ax.set_ylabel(metric.upper(), fontsize=AXISSIZE)
            ax.set_ylim(0, 1)
            ax.set_title(f'Megadetector {metric} With and Without Classifier')
        else: 
            ax.set_ylim(0, 3500)
            ax.set_ylabel("Inference Time (ms)", fontsize=AXISSIZE)
            ax.set_title(f'Megadetector Inference Time With and Without Classifier')
        
        plt.tight_layout()
        plt.savefig(save_path + f"_{metric}.png")

def plot_process_time(df, deployment_time = 6, max_fps = 3, save_path = os.path.join("ObjectDetection/Megadetector/Results/plots","mega_process_time.png")):
    versions = df["version"].unique()

    accs = {True:[],False:[]}
    for version in versions:
        for classifier in [True,False]:
            filtered_df = df[(df['version'] == version) & (df['classify'] == classifier)]
            accs[classifier].append(filtered_df.iloc[0]['mAP50'])
    
    times = [round(num, 3) for num in accs[True]]
    no_images = deployment_time * 60 * 60

    times_fsp = {}
    for i in range(1,max_fps + 1):
        times_fsp[f"{i} FPS"] = [round(num * no_images * i / 60 / 60,2) for num in times]

    x = np.arange(len(versions))
    width = BARWIDTH
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=FIGSIZE)

    for idx, (attribute, measurement) in enumerate(times_fsp.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
        ax.bar_label(rects, padding=2, fontsize=TICKSIZE)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Process Time (Hours)', fontsize=AXISSIZE)
    ax.set_xlabel('MegaDetector Versions', fontsize=AXISSIZE)
    ax.set_title(f'Megadetector 6 Hour Deployment Processing Time on PC', fontsize=TITLESIZE)
    ax.set_xticks(x + width / 2, versions)
    ax.legend(loc='upper left', ncols=2)

    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == "__main__":
    if os.path.exists(PLOT_PATH): shutil.rmtree(PLOT_PATH)
    os.mkdir(PLOT_PATH)

    results_df = pd.read_csv(os.path.join(RES_PATH,"results.csv"))

    plot_accuracies(results_df)
    plot_process_time(results_df)
