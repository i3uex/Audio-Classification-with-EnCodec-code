import numpy as np
import torchaudio
import torch.nn.functional as F


def __load_function_48khz():
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    # Instantiate a pretrained EnCodec model & move model to GPU
    encodec_model = EncodecModel.encodec_model_48khz()
    encodec_model = encodec_model.to("cuda:0")

    def load_function(path):
        print(path)
        wav, sr = torchaudio.load(path)
        wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
        wav = wav.unsqueeze(0)
        encoded_frames = []
        for offset in range(0, wav.shape[-1], encodec_model.segment_stride):
            frame = wav[:, :, offset: offset + encodec_model.segment_length]
            frame = F.pad(frame, (0, encodec_model.segment_length - frame.shape[-1]), "constant", 0)
            encoded_frames.append(encodec_model.encoder(frame.to("cuda:0")).detach().cpu().numpy()[0])
        return np.array(encoded_frames)

    return load_function


def __load_function_24khz():
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    # Instantiate a pretrained EnCodec model & move model to GPU
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model = encodec_model.to("cuda:0")

    def load_function(path, window_size=1):
        print(path)
        wav, sr = torchaudio.load(path)
        wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
        wav = wav.unsqueeze(0)
        encoded_frames = []
        win = encodec_model.sample_rate*window_size
        for offset in range(0, wav.shape[-1], win):
            frame = wav[:, :, offset: offset + win]
            frame = F.pad(frame, (0, win - frame.shape[-1]), "constant", 0)
            encoded_frames.append(encodec_model.encoder(frame.to("cuda:0")).detach().cpu().numpy()[0])
        return np.array(encoded_frames)

    return load_function


def __load_function_melspectrogram():
    import librosa

    def load_function(path, window_size=1):
        print(path)
        wav, sr = librosa.load(path, sr=22050, mono=True)
        win = sr * window_size
        encoded_frames = []
        for offset in range(0, wav.shape[-1], win):
            frame = wav[offset: offset + win]
            frame = np.pad(frame, (0, win - frame.shape[-1]), "constant")
            encoded_frames.append(librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=128, n_fft=512, hop_length=128))
        return np.array(encoded_frames)
    return load_function


def get_load_function(model_name):
    return {
        "48khz": __load_function_48khz,
        "24khz": __load_function_24khz,
        "melspectrogram": __load_function_melspectrogram,
    }[model_name]()