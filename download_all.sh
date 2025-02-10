#!/bin/bash
# GTZAN Speech_Music dataset
curl -L -o gtzan-musicspeech-collection.zip https://www.kaggle.com/api/v1/datasets/download/lnicalo/gtzan-musicspeech-collection
unzip gtzan-musicspeech-collection.zip -d "datasets/GTZAN Speech_Music"
rm gtzan-musicspeech-collection.zip
mv datasets/GTZAN\ Speech_Music/speech_wav datasets/GTZAN\ Speech_Music/speech
mv datasets/GTZAN\ Speech_Music/music_wav datasets/GTZAN\ Speech_Music/music

# GTZAN Genre Dataset
curl -L -o gtzan-genre-collection.zip  https://www.kaggle.com/api/v1/datasets/download/carlthome/gtzan-genre-collection
unzip gtzan-genre-collection.zip -d "datasets/GTZAN Genre"
rm gtzan-genre-collection.zip

# ESC-50 Dataset
curl -L -o ESC-50.zip  https://github.com/karoldvl/ESC-50/archive/master.zip
unzip ESC-50.zip -d "datasets/"
rm ESC-50.zip
python3 datasets/parse_ESC50_meta.py