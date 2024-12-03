import torch
import torchaudio.transforms as T
from torch import nn


def get_audio_transforms():
    """
    Audio preprocessing transformations
    """
    return nn.Sequential(
        T.MelSpectrogram(sample_rate=config.SAMPLE_RATE),
        T.AmplitudeToDB()
    )