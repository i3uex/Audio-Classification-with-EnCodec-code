from utils import perform_training

datasets = ["genres"]
models = ["48khz", "24khz", "melspectrogram"]

for dataset in datasets:
    for model in models:
        perform_training(dataset, model)
