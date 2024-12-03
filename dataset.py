import os

import librosa
import torch
from torch.utils.data import Dataset


class PetDataset(Dataset):
    def __init__(self, data_dir, labels_dict=None, transform=None,
                 target_sample_rate=16000, num_samples=16000):
        self.data_dir = data_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.labels_dict = labels_dict or {}

        # Get list of audio files with their descriptions
        self.audio_files = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.mp3'):
                description = self.labels_dict.get(filename, "Unknown pet behavior")
                self.audio_files.append((filename, description))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file, description = self.audio_files[idx]
        audio_path = os.path.join(self.data_dir, audio_file)

        # Load audio file
        signal, sr = librosa.load(audio_path, sr=self.target_sample_rate)

        # Extract audio features
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_processed = mfcc.T  # Transpose to match model input shape

        signal_tensor = torch.from_numpy(mfcc_processed).float()

        # Truncate or pad to consistent length
        if len(signal_tensor) > self.num_samples:
            signal_tensor = signal_tensor[:self.num_samples]
        else:
            pad_width = self.num_samples - len(signal_tensor)
            signal_tensor = torch.nn.functional.pad(signal_tensor, (0, 0, 0, pad_width))

        return signal_tensor, description
