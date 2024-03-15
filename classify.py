from utils import perform_training

datasets = [
    "speech_music",
    "genres",
    "env",
]
models = ["melspectrogram", "24khz", "48khz"]

for dataset in datasets:
    for model in models:
        perform_training(dataset, model)
