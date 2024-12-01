import torchaudio.transforms as T


def get_transforms():
    return T.Resample(orig_freq=48000, new_freq=16000)
