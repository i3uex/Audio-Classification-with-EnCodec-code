import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from loaders import get_load_function

DATASETS_BASE_PATH = "/path/to/datasets"
DATASET2BASE_FOLDER = {
    "genres": f"{DATASETS_BASE_PATH}/GTZAN Genres",
    "speech_music": f"{DATASETS_BASE_PATH}/GTZAN Speech_Music",
}

DATASET2CLASSES = {
    "genres": ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
    "speech_music": ["speech", "music"],
}


def shuffle(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    return zip(*c)


def load_dataset(dataset_name, model_name, classes, max_items_per_folder=100):
    dataset_base_folder = DATASET2BASE_FOLDER[dataset_name]
    load_function = get_load_function(model_name)
    X_file_path = f"X_{dataset_name}_{model_name}.npy"
    Y_file_path = f"Y_{dataset_name}_{model_name}.npy"
    if os.path.exists(X_file_path) and os.path.exists(Y_file_path):
        X = np.load(X_file_path).astype(np.float32)
        Y = np.load(Y_file_path).astype(np.float32)
    else:
        Y = []
        X = None
        for klass in classes:
            glob_str = f"{dataset_base_folder}/{klass}/*"
            _X = [load_function(path) for path in glob.glob(glob_str)[:max_items_per_folder]]
            if len(_X) == 0:
                print(f"No files found for {klass} in {glob_str}")
                continue
            if X is None:
                X = np.concatenate(_X)
            else:
                X = np.concatenate([X, np.concatenate(_X)])
            Y.extend([[1. if klass == x else 0. for x in classes]]*(len(X)-len(Y)))
        Y = np.array(Y, dtype=np.float32)
        X = np.array(X)
        np.save(X_file_path, X)
        np.save(Y_file_path, Y)
    return X, Y


def get_MNIST_train_model(classes, feature_len=128, kernel_size=5):
    return torch.nn.Sequential(
        torch.nn.Conv1d(128, feature_len, kernel_size),
        torch.nn.MaxPool1d(kernel_size),
        torch.nn.ReLU(),
        torch.nn.Conv1d(128, feature_len*2, kernel_size),
        torch.nn.Dropout1d(),
        torch.nn.MaxPool1d(kernel_size),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(classes)
    ).to("cuda:0")


def train_loop(X, Y, classification_model, optimizer=torch.optim.SGD, lr=0.01, criterion=torch.nn.CrossEntropyLoss, epochs=500, val_size=0.2, batch_size=256):
    criterion = criterion()
    optimizer = optimizer(classification_model.parameters(), lr=lr)
    val_size = int(len(X)*val_size)
    X, Y = shuffle(X, Y)
    train_set = list(zip(X[:-val_size], Y[:-val_size]))
    val_set = list(zip(X[-val_size:], Y[-val_size:]))
    train_loader = DataLoader(dataset=train_set, batch_size = batch_size)
    val_loader = DataLoader(dataset=val_set, batch_size = batch_size)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        correct = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = classification_model(inputs.to("cuda:0")).cpu()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == torch.max(labels.data, 1)[1]).sum()
        t_loss = loss.item()
        train_loss.append(t_loss)
        train_accuracy = 100 * (correct.item()) / len(train_set)
        train_acc.append(train_accuracy)
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = classification_model(inputs.to("cuda:0")).cpu()
                _loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == torch.max(labels.data, 1)[1]).sum()
        v_loss = _loss.item()
        val_loss.append(v_loss)
        val_accuracy = 100 * (correct.item()) / len(val_set)
        val_acc.append(val_accuracy)
        if epoch % 10 == 0:
            print(f'{epoch}/{epochs} - Train Loss: {t_loss}. Train Accuracy: {train_accuracy}. Val Loss: {v_loss}. Val Accuracy: {val_accuracy}')
    return train_loss, val_loss, train_acc, val_acc


def perform_training(dataset_name, model_name, results_folder="results"):
    classes = DATASET2CLASSES[dataset_name]
    X, Y = load_dataset(dataset_name, model_name, classes)
    train_loss, val_loss, train_acc, val_acc = train_loop(X, Y, get_MNIST_train_model(len(classes)))
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    with open(f"{results_folder}/classify_{dataset_name}_{model_name}_train_acc.json", "w") as f:
        json.dump(train_acc, f)
    with open(f"{results_folder}/classify_{dataset_name}_{model_name}_train_loss.json", "w") as f:
        json.dump(train_loss, f)
    with open(f"{results_folder}/classify_{dataset_name}_{model_name}_val_acc.json", "w") as f:
        json.dump(val_acc, f)
    with open(f"{results_folder}/classify_{dataset_name}_{model_name}_val_loss.json", "w") as f:
        json.dump(val_loss, f)