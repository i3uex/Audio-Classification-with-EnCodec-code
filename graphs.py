import json
import matplotlib.pyplot as plt

dataset_name = "speech_music"
result_folder = "results"
for model_name in ["melspectrogram", "24khz", "48khz"]:
    with open(f"{result_folder}/classify_{dataset_name}_{model_name}_train_acc.json") as f:
        train_acc = json.load(f)
    with open(f"{result_folder}/classify_{dataset_name}_{model_name}_val_acc.json") as f:
        val_acc = json.load(f)
    with open(f"{result_folder}/classify_{dataset_name}_{model_name}_train_loss.json") as f:
        train_loss = json.load(f)
    with open(f"{result_folder}/classify_{dataset_name}_{model_name}_val_loss.json") as f:
        val_loss = json.load(f)
    plt.plot(train_acc, label=model_name + " train Accuracy")
    plt.plot(val_acc, label=model_name + " validation Accuracy")
plt.legend()
plt.title(f"{dataset_name} Accuracy")
plt.show()