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

def plot_model_accuracy(df, save_path = os.path.join(PLOT_DIR, "class_test_accuracies.png"), colours=COLOURS):
    model_names = df['model_name'].unique().tolist()
    pretrained = [] 
    no_pretrained = []

    for model_name in model_names:
        filtered_df = df[df['model_name'] == model_name]
        pretrained.append(filtered_df[filtered_df['pretrained'] == True]['test_acc'].iloc[0])
        no_pretrained.append(filtered_df[filtered_df['pretrained'] == False]['test_acc'].iloc[0])

    sorted_pairs = sorted(zip(model_names, pretrained, no_pretrained), key=lambda x: x[1])

    model_names, pretrained, no_pretrained = zip(*sorted_pairs)

    test_accuracies = {
        'Pretrained Weights': list(pretrained),
        'Random Weights': list(no_pretrained)
    }

    x = np.arange(len(model_names))
    width = 0.25 
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for idx, (attribute, measurement) in enumerate(test_accuracies.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colours[idx])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracies of Pretrained and Random Weighst Across Models')
    ax.set_xticks(x + width / 2, model_names)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 100)  # Assuming accuracy is between 0 and 1

    plt.savefig(save_path)

def plot_inference_times(df, save_path=os.path.join(PLOT_DIR, "class_inference_times.png")):
    # Filter df for pretrained models
    df = df[df['pretrained'] == True]
    
    # Sort the DataFrame by 'inference' in ascending order
    df = df.sort_values(by='inference', ascending=True)

    # Get values after sorting
    models = df['model_name'].tolist()
    inference = df['inference'].tolist()
    
    fig = plt.figure(figsize=(10, 5))

    # Creating the bar plot with sorted values
    plt.bar(models, inference, color=COLOURS[:len(inference)], width=0.3)

    plt.xlabel("Model")
    plt.ylabel("Inference Time")
    plt.title("Model Architecture Inference Times")
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
    plt.tight_layout()  # Adjust layout to fit labels
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memo

if os.path.exists(PLOT_DIR):
    shutil.rmtree(PLOT_DIR)
os.mkdir(PLOT_DIR)

# plot the binary model accuracies
plot_model_accuracy(results_df)
plot_inference_times(results_df)