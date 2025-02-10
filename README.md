# Beyond Spectrograms: Rethinking Audio Classification from EnCodec’s Latent Space
Code for the paper "Beyond Spectrograms: Rethinking Audio Classification from EnCodec’s Latent Space"

# Running the Experiments

In order to reproduce the results of the paper, you need to follow the steps below:

## 1. Downloading datasets

All the datasets must be located in the `datasets` folder. This folder should contain the following subfolders after downloading the datasets:

- GTZAN Speech_Music: Contains the GTZAN Speech Music dataset. Class folders should be named "speech" and "music".
- GTZAN Genre: Contains the GTZAN Music Genre dataset. Class folders should be named according to the genre and be located in the "genres" folder.
- ESC-50: Contains the ESC-50 dataset. Class folders should be named according to the class and be located in the "classes" folder. In order to parse the dataset, you should run the following script:

```bash
#!/bin/bash
python3 parse_ESC50_meta.py
```

In order to download and parse all datasets, you can run the following script:

```bash
#!/bin/bash
./download_all.sh
```

Else, you can download the datasets manually:

### 1.1. GTZAN Speech music dataset

Can be downloaded from [here](https://www.kaggle.com/datasets/lnicalo/gtzan-musicspeech-collection) or by running the following script:

```
#!/bin/bash
curl -L -o gtzan-musicspeech-collection.zip https://www.kaggle.com/api/v1/datasets/download/lnicalo/gtzan-musicspeech-collection
unzip gtzan-musicspeech-collection.zip -d "datasets/GTZAN Speech_Music"
rm gtzan-musicspeech-collection.zip
```

Then you should rename the class folders from "speech_wav" and "music_wav" to "speech" and "music" respectively. You can do this by running:

```
#!/bin/bash
mv datasets/GTZAN\ Speech_Music/speech_wav datasets/GTZAN\ Speech_Music/speech
mv datasets/GTZAN\ Speech_Music/music_wav datasets/GTZAN\ Speech_Music/music
```


### 1.2. GTZAN Music genre dataset

Can be downloaded from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) or by running the following script:

```
#!/bin/bash
curl -L -o gtzan-dataset-music-genre-classification.zip https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d "datasets/GTZAN Genre"
rm gtzan-dataset-music-genre-classification.zip
```


### 1.3. Environmental Sound Classification (ESC-50)
Can be downloaded from [here](https://github.com/karolpiczak/ESC-50) or by running the following script:

```
curl -L -o ESC-50.zip  https://github.com/karoldvl/ESC-50/archive/master.zip
unzip ESC-50.zip -d "datasets/"
rm ESC-50.zip
```

After downloading the dataset, you should run the following script to parse the metadata:

```bash
python3 datasets/parse_ESC50_meta.py
```

## 2. Preparing the environment

You can create a virtual environment and install the dependencies by running the following commands:

### 2.1. Create a virtual environment

```bash
virtualenv venv
source venv/bin/activate
```

### 2.2. Install the dependencies

```bash
pip install -r requirements.txt
```

## 3. Training the models

In order to train the models, you can run the following script:

```bash
python3 classify.py
```

