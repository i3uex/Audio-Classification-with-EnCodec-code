import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from loaders import get_loader_model

DATASETS_BASE_PATH = "/path/to/datasets"
DATASET2BASE_FOLDER = {
    "genres": f"{DATASETS_BASE_PATH}/GTZAN Genres",
    "speech_music": f"{DATASETS_BASE_PATH}/GTZAN Speech_Music",
    "env": f"{DATASETS_BASE_PATH}/ESC-50-master/classes",
}

DATASET2CLASSES = {
    "env": ['airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn', 'cat', 'chainsaw', 'chirping_birds', 'church_bells', 'clapping', 'clock_alarm', 'clock_tick', 'coughing', 'cow', 'crackling_fire', 'crickets', 'crow', 'crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock', 'drinking_sipping', 'engine', 'fireworks', 'footsteps', 'frog', 'glass_breaking', 'hand_saw', 'helicopter', 'hen', 'insects', 'keyboard_typing', 'laughing', 'mouse_click', 'pig', 'pouring_water', 'rain', 'rooster', 'sea_waves', 'sheep', 'siren', 'sneezing', 'snoring', 'thunderstorm', 'toilet_flush', 'train', 'vacuum_cleaner', 'washing_machine', 'water_drops', 'wind'],
    "genres": ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
    "speech_music": ["speech", "music"],
}
WINDOW_SECONDS = 1
STEP_SECONDS = 1


def shuffle(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    return zip(*c)


def load_dataset(dataset_name, model_name, classes, max_items_per_folder=100, win_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS):
    dataset_base_folder = DATASET2BASE_FOLDER[dataset_name]
    loader_model = get_loader_model(model_name, win_seconds, step_seconds)
    X_file_path = f"X_{dataset_name}_{model_name}_win{win_seconds}_step{step_seconds}.npy"
    Y_file_path = f"Y_{dataset_name}_{model_name}_win{win_seconds}_step{step_seconds}.npy"
    if os.path.exists(X_file_path) and os.path.exists(Y_file_path):
        X = np.load(X_file_path).astype(np.float32)
        Y = np.load(Y_file_path).astype(np.float32)
    else:
        Y = []
        X = None
        for klass in classes:
            glob_str = f"{dataset_base_folder}/{klass}/*"
            _X = [loader_model.load(path) for path in glob.glob(glob_str)[:max_items_per_folder]]
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


def get_MNIST_train_model(classes, channels=128, feature_len=128, kernel_size=5, max_pool_size=5):
    return torch.nn.Sequential(
        torch.nn.Conv1d(channels, feature_len, kernel_size),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(max_pool_size),
        torch.nn.Conv1d(feature_len, feature_len*2, kernel_size),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(max_pool_size),
        torch.nn.Dropout(),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(classes)
    ).to("cuda:0")


def check_step(loader, classification_model, criterion):
    correct = 0
    for inputs, labels in loader:
        outputs = classification_model(inputs.to("cuda:0")).cpu()
        _loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == torch.max(labels.data, 1)[1]).sum()
    return correct, _loss


def train_loop(X, Y, classification_model, optimizer=torch.optim.SGD, lr=0.01, criterion=torch.nn.CrossEntropyLoss, epochs=500, val_size=0.2, test_size=0.2, batch_size=256):
    criterion = criterion()
    optimizer = optimizer(classification_model.parameters(), lr=lr)
    train_size = 1-val_size-test_size
    val_index = int(len(X) * train_size)
    test_index = int(len(X) * (train_size+val_size))
    X, Y = shuffle(X, Y)
    train_set = list(zip(X[:val_index], Y[:val_index]))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size)
    val_set = list(zip(X[val_index:test_index], Y[val_index:test_index]))
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size)
    test_set = list(zip(X[test_index:], Y[test_index:]))
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    best_val_loss = float("inf")
    best_model = None
    # Perform training
    for epoch in range(epochs):
        correct = 0
        # Iterate over train set
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = classification_model(inputs.to("cuda:0")).cpu()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == torch.max(labels.data, 1)[1]).sum()
        # Compute train metrics
        t_loss = loss.item()
        all_train_loss.append(t_loss)
        train_accuracy = 100 * (correct.item()) / len(train_set)
        all_train_acc.append(train_accuracy)
        correct = 0
        # Compute validation accuracy and loss and save best model
        with torch.no_grad():
            correct, _loss = check_step(val_loader, classification_model, criterion)
        # Compute validation metrics
        v_loss = _loss.item()
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_model = classification_model.state_dict()
        all_val_loss.append(v_loss)
        val_accuracy = 100 * (correct.item()) / len(val_set)
        all_val_acc.append(val_accuracy)
        if epoch % 10 == 0:
            print(f'{epoch}/{epochs} - Train Loss: {t_loss}. Train Accuracy: {train_accuracy}. Val Loss: {v_loss}. Val Accuracy: {val_accuracy}')

    classification_model.load_state_dict(best_model)
    with torch.no_grad():
        correct, _loss = check_step(test_loader, classification_model, criterion)
    t_loss = _loss.item()
    test_acc = 100 * (correct.item()) / len(test_set)

    print(f'Test Loss: {t_loss}. Test Accuracy: {test_acc}')
    return all_train_loss, all_val_loss, t_loss, all_train_acc, all_val_acc, test_acc, best_model


def perform_training(dataset_name, model_name, results_folder="results"):
    classes = DATASET2CLASSES[dataset_name]
    X, Y = load_dataset(dataset_name, model_name, classes)
    train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, best_model = train_loop(X, Y, get_MNIST_train_model(len(classes)))
    torch.save(best_model, f"classify_{dataset_name}_{model_name}_model.pth")
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    with open(f"{results_folder}/classify_{dataset_name}_{model_name}.json", "w") as f:
        json.dump(
            {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            }, f)