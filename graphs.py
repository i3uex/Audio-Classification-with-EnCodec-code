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
    for model_name in models:
        print("----", model_name, "for dataset:"
                                  "", dataset_name, "----")
        with open(f"{result_folder}/classify_{dataset_name}_{model_name}.json") as f:
            data = json.load(f)

        if GRAPH_TRAINING_DATA:
            plt.plot(smooth(data['train_acc'], 0.8), label=model_name + " train Accuracy")
        plt.plot(smooth(data['val_acc'], 0.8), label=model_name + " validation Accuracy")
        print("Train Loss", np.min(data['train_loss']))
        print("Train Accuracy", np.max(data['train_acc']))
        print("Validation Loss", np.min(data['val_loss']))
        print("Validation Accuracy", np.max(data['val_acc']))
        print("Test Loss", data['test_loss'])
        print("Test Accuracy", data['test_acc'])
        print("---------------------------------------------------")
    if GRAPH_LEGEND:
        plt.legend()
    if GRAPH_TITLE:
        plt.title(f"{dataset_name} Accuracy")
    plt.show()
