import abc
import numpy as np
import torchaudio
import torch.nn.functional as F
import librosa
from encodec.utils import convert_audio


def np_window(x: np.array, window_seconds=2, step_seconds=1, sr=22050) -> np.ndarray:
    """
    Apply a window to the audio
    :param x: Audio in np array format
    :param window_seconds: seconds of the window
    :param step_seconds: seconds of the step
    :param sr: sample rate of the audio
    :return: windowed audio
    """
    window_frames = int(window_seconds * sr)
    step_frames = int(step_seconds * sr)
    shape = (x.size - window_frames + 1, window_frames)
    strides = x.strides * 2  # * window ??
    return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape, writeable=False)[0::step_frames]


def torch_window(x: np.array, window_seconds=2, step_seconds=1, sr=22050) -> np.ndarray:
    """
    Apply a window to the audio
    :param x: Audio in np array format
    :param window_seconds: seconds of the window
    :param step_seconds: seconds of the step
    :param sr: sample rate of the audio
    :return: windowed audio
    """
    window_frames = int(window_seconds * sr)
    step_frames = int(step_seconds * sr)
    return x.unfold(-1, window_frames, step_frames)


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
    def __init__(self, window_size=1, step_size=1):
        self.window_size = window_size
        self.step_size = step_size

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
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels).unsqueeze(0)
        windowed_items = torch_window(wav, sr=self.model.sample_rate, window_seconds=self.window_size, step_seconds=self.step_size).transpose(0, 2).transpose(1, 2)
        return np.array([self.encoder(item.to("cuda:0")).detach().cpu()[0] for item in windowed_items])


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
        encoded_frames = []
        windowed_items = np_window(wav, sr=sr, window_seconds=self.window_size, step_seconds=self.step_size)
        for frame in windowed_items:
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


def get_loader_model(model_name, win, step):
    if model_name in ["24khz", "48khz"]:
        return AudioClassifierModelEncodec(model_name)
    elif model_name == "melspectrogram":
        return AudioClassifierModelLibrosa()
    else:
        raise ValueError(f"Invalid model name: {model_name}")
