import abc
import numpy as np
import torchaudio
import torch.nn.functional as F
import librosa
from encodec.utils import convert_audio


def load_encodec_model(model_name):
    from encodec import EncodecModel

    def load24khz():
        return EncodecModel.encodec_model_24khz()

    def load48khz():
        return EncodecModel.encodec_model_48khz()

    if model_name == "24khz":
        return load24khz()
    elif model_name == "48khz":
        return load48khz()
    else:
        raise ValueError(f"Invalid model name: {model_name}")


class AudioClassifierModelBase(metaclass=abc.ABCMeta):
    def __init__(self, window_size=1):
        self.window_size = window_size

    @abc.abstractmethod
    def load(self, path):
        pass


class AudioClassifierModelEncodecBase(AudioClassifierModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.encoder = None

    def load(self, path):
        print(path)
        wav, sr = torchaudio.load(path)
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        wav = wav.unsqueeze(0)
        encoded_frames = []
        win = self.model.sample_rate * self.window_size
        for offset in range(0, wav.shape[-1], win):
            frame = wav[:, :, offset: offset + win]
            frame = F.pad(frame, (0, win - frame.shape[-1]), "constant", 0)
            encoded_frames.append(self.encoder(frame.to("cuda:0")).detach().cpu().numpy()[0])
        return np.array(encoded_frames)


class AudioClassifierModelEncodec(AudioClassifierModelEncodecBase):

    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model = load_encodec_model(model_name).to("cuda:0")
        self.encoder = self.model.encoder


class AudioClassifierModelLibrosa(AudioClassifierModelBase):

    def __init__(self, n_mels=128, n_fft=512, hop_length=128, **kwargs):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load(self, path):
        print(path)
        wav, sr = librosa.load(path, mono=True)
        win = int(sr * self.window_size)
        encoded_frames = []
        for offset in range(0, wav.shape[-1], win):
            frame = wav[offset: offset + win]
            frame = np.pad(frame, (0, win - frame.shape[-1]), "constant")
            encoded_frames.append(
                librosa.feature.melspectrogram(
                    y=frame,
                    sr=sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
            )
        return np.array(encoded_frames)


def get_loader_model(model_name):
    if model_name in ["24khz", "48khz"]:
        return AudioClassifierModelEncodec(model_name)
    elif model_name == "melspectrogram":
        return AudioClassifierModelLibrosa()
    else:
        raise ValueError(f"Invalid model name: {model_name}")
