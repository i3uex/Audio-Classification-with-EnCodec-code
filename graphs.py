import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt

GRAPH_TITLE = True
GRAPH_LEGEND = True
GRAPH_TRAINING_DATA = False
datasets = [
    "speech_music",
    "genres",
    "env"
]
result_folder = "results"
models = ["melspectrogram", "24khz", "48khz"]



def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


for dataset_name in datasets:
    ax = plt.gca()
    for i, model_name in enumerate(models):
        print("----", model_name, "for dataset:", dataset_name, "----")
        with open(f"{result_folder}/classify_{dataset_name}_{model_name}.json") as f:
            data = json.load(f)
        if GRAPH_TRAINING_DATA:
            ax.plot(smooth(data['train_acc'], 0.8), label=model_name + " train Accuracy")
        ax.plot(smooth(data['val_acc'], 0.8), label=model_name + (" validation Accuracy" if GRAPH_TRAINING_DATA else ""))
        print("Test Loss", data['test_loss'])
        print("Test Accuracy", data['test_acc'])
        if i == 2:  # 1 for 24khz, 2 for 48khz
            # Get epoch where melspectrogram accuracy is already better than max
            epc = np.argwhere(data['val_acc'] > desired_acc)[0][0]
            # Annotate the point
            ax.annotate(f'{round(data["val_acc"][epc], 2)}%: Already better than max\nmelspectrogram accuracy ({round(desired_acc, 2)}%)\nat epoch {epc}',
                            xy=(epc, desired_acc),
                            xycoords='data',
                            xytext=(0.5, 0.5),
                            textcoords='figure fraction',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                        arrowprops=dict(facecolor='black', shrink=0.125, width=1, headwidth=7.5))
            ax.scatter(epc, desired_acc, color="red", marker="x", s=100)
        # Get mel spectrogram max accuracy and loss
        elif i == 0:
            desired_acc = np.max(data['val_acc'])
    # Plot max melspectrogram accuracy
    ax.plot([desired_acc]*len(data['val_acc']), label="Max melspectrogram accuracy", linestyle="--", color="gray")
    if GRAPH_LEGEND:
        ax.legend()
    if GRAPH_TITLE:
        plt.title(f"{dataset_name} Accuracy")
    ax.set_ylim(0, 100)
    plt.show()
    print("---------------------------------------------------")
