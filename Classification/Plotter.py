import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
import numpy as np

BINARY_RESULTS_DIR = os.path.join("Classification", "Results", "Binary")
PLOT_DIR = os.path.join(BINARY_RESULTS_DIR, "plots")

results_df = pd.read_csv(os.path.join(BINARY_RESULTS_DIR, "results.csv"))
COLOURS=[x for i,x in enumerate(matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, 10))) if i!=1]
FIGSIZE = (8,6)
TITLESIZE = 15
AXISSIZE = 12
TICKSIZE = 10
BARWIDTH = 0.3

def plot_model_accuracy(df, save_path = os.path.join(PLOT_DIR, "class_test_accuracies.png"), colours=COLOURS):
    
    # Filter df for pretrained models
    df = df[df['pretrained'] == True]
    
    # Sort the DataFrame by 'inference' in ascending order
    df = df.sort_values(by='test_acc', ascending=True)

    # Get values after sorting
    models = df['model_name'].tolist()
    inference = [num for num in df['test_acc'].tolist()]
    
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Create a bar plot with sorted values and save the bar container
    bars = ax.bar(models, inference, color=COLOURS[:len(inference)], width=BARWIDTH)

    # Add bar labels using `ax.bar_label`
    ax.bar_label(bars, labels=[f"{val:.0f}" for val in inference], fontsize=TICKSIZE, padding=3)

    # Set labels and title
    ax.set_ylabel('Test Accuracy (%)', fontsize=AXISSIZE)
    plt.xlabel("Model Architecture", fontsize=AXISSIZE)
    ax.set_title('Test Accuracies Across Model Architectures', fontsize=TITLESIZE)
    
    # Rotate x labels for better visibility
    plt.xticks(ha='right')  
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(save_path)
    plt.close(fig) 

def plot_inference_times(df, save_path=os.path.join(PLOT_DIR, "class_inference_times.png")):
    # Filter df for pretrained models
    df = df[df['pretrained'] == True]
    
    # Sort the DataFrame by 'inference' in ascending order
    df = df.sort_values(by='inference', ascending=True)

    # Get values after sorting
    models = df['model_name'].tolist()
    inference = [num * 1000 for num in df['inference'].tolist()]
    
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Create a bar plot with sorted values and save the bar container
    bars = ax.bar(models, inference, color=COLOURS[:len(inference)], width=BARWIDTH)

    # Add bar labels using `ax.bar_label`
    ax.bar_label(bars, labels=[f"{val:.2f}" for val in inference], fontsize=TICKSIZE, padding=3)

    # Set labels and title
    ax.set_xlabel("Model Architecture", fontsize=AXISSIZE)
    ax.set_ylabel("Inference Time (ms)", fontsize=AXISSIZE)
    ax.set_title("Model Architecture Inference Times", fontsize=TITLESIZE)
    
    # Rotate x labels for better visibility
    plt.xticks(ha='right')  
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(save_path)
    plt.close(fig) 

if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.mkdir(PLOT_DIR)

# plot the binary model accuracies
plot_model_accuracy(results_df)
plot_inference_times(results_df)