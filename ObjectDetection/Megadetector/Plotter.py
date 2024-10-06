
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

def plot_accuracies(df, save_path = os.path.join(PLOT_PATH,"mega")):
    metrics = ["mAP50", "f1", "inference"]

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

        fig, ax = plt.subplots(layout='constrained')

        for idx, (attribute, measurement) in enumerate(test_accuracies.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
            ax.bar_label(rects, padding=2)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xticks(x + width / 2, versions)
        ax.legend(loc='upper left', ncols=2)
        ax.set_xlabel("Mega Version")
        if metric != "inference": 
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.set_title(f'Megadetector {metric} With and Without Classifier')
        else: 
            ax.set_ylim(0, 3500)
            ax.set_ylabel("Inference Time (ms)")
            ax.set_title(f'Megadetector Inference Time With and Without Classifier')

        plt.savefig(save_path + f"_{metric}.png")

def plot_inference_time(df, save_path = os.path.join(PLOT_PATH,"accuracies.png")):
    versions = df["version"].unique()
    print(versions)
    accs = {True:[],False:[]}
    for version in versions:
        for classifier in [True,False]:
            filtered_df = df[(df['version'] == version) & (df['classify'] == classifier)]
            accs[classifier].append(filtered_df.iloc[0]['mAP50'])
    
    test_accuracies = {
            'Classifier': list([round(num, 3) for num in accs[True]]),
            'NoClassifier': list([round(num, 3) for num in accs[False]])
        }

    x = np.arange(len(versions))
    width = 0.4
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for idx, (attribute, measurement) in enumerate(test_accuracies.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=COLOURS[idx])
        ax.bar_label(rects, padding=2)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mAP50')
    ax.set_title(f'Mega a and b mAP50 with and without classifier')
    ax.set_xticks(x + width / 2, versions)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 1)

    plt.savefig(save_path)


if __name__ == "__main__":
    if os.path.exists(PLOT_PATH): shutil.rmtree(PLOT_PATH)
    os.mkdir(PLOT_PATH)

    results_df = pd.read_csv(os.path.join(RES_PATH,"results.csv"))

    plot_accuracies(results_df)
