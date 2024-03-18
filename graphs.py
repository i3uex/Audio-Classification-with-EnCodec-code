import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt

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
        with open(f"{result_folder}/classify_{dataset_name}_{model_name}_train_acc.json") as f:
            train_acc = json.load(f)
        with open(f"{result_folder}/classify_{dataset_name}_{model_name}_val_acc.json") as f:
            val_acc = json.load(f)
        with open(f"{result_folder}/classify_{dataset_name}_{model_name}_train_loss.json") as f:
            train_loss = json.load(f)
        with open(f"{result_folder}/classify_{dataset_name}_{model_name}_val_loss.json") as f:
            val_loss = json.load(f)
        plt.plot(smooth(train_acc, 0.8), label=model_name + " train Accuracy")
        plt.plot(smooth(val_acc, 0.8), label=model_name + " validation Accuracy")
        print("Train Loss", np.min(val_loss))
        print("Train Accuracy", np.max(val_acc))
        print("Validation Loss", np.min(val_loss))
        print("Validation Accuracy", np.max(val_acc))
        print("---------------------------------------------------")
    plt.legend()
    plt.title(f"{dataset_name} Accuracy")
    plt.show()