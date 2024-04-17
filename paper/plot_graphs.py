import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from encodec import EncodecModel

start_second = 1

for _file in ['brahms', 'libri1']:
    EXAMPLE_FILE = librosa.ex(_file)
    y, sr = librosa.load(EXAMPLE_FILE)
    y = y[start_second*sr:start_second*sr+sr]

    # Generate mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=512, hop_length=128)
    plt.imshow(mel, interpolation='nearest', aspect='auto')
    plt.title("Mel-frequency spectrogram")
    plt.show()

    # Convert mel spectrogram to dB for better visualization
    S_dB = librosa.power_to_db(mel, ref=np.max)
    plt.imshow(S_dB, interpolation='nearest', aspect='auto')
    plt.title("Mel-frequency spectrogram in dB")
    plt.show()

    # Generate encodec 48Khz featuremap
    encodec_model_48khz = EncodecModel.encodec_model_48khz()
    frame = librosa.resample(y, orig_sr=sr, target_sr=encodec_model_48khz.sample_rate)
    frame = torch.tensor([frame, frame]).unsqueeze(0)
    E = encodec_model_48khz.encoder(frame).detach().numpy()[0]

    plt.title("Encodec 48Khz featuremap")
    plt.imshow(E, interpolation='nearest', aspect='auto')
    plt.show()

    # Generate encodec 24Khz featuremap
    encodec_model_24khz = EncodecModel.encodec_model_24khz()

    frame = librosa.resample(y, orig_sr=sr, target_sr=encodec_model_24khz.sample_rate)
    frame = torch.tensor([frame]).unsqueeze(0)
    E = encodec_model_24khz.encoder(frame).detach().numpy()[0]

    plt.title("Encodec 24Khz featuremap")
    plt.imshow(E, interpolation='nearest', aspect='auto')
    plt.show()
