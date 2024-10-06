import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

results_df = pd.read_csv(os.path.join("Classification", "Results", "Binary", "results.csv"))

def plot_model_accuracy(df, save_path = os.path.join("Classification","Results","Binary", "plots", "test_accuracies.png"), colours=matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, 10))):
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
        'Pretrained': list(pretrained),
        'Untrained': list(no_pretrained)
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
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracies of Pretrained and Untrained Models')
    ax.set_xticks(x + width / 2, model_names)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 100)  # Assuming accuracy is between 0 and 1

    plt.savefig(save_path)

# plot the binary model accuracies
plot_model_accuracy(results_df)