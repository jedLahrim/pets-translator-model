import os

import librosa
import torch
import torchaudio.transforms
from torch.utils.data import Dataset

# Check available backends
print(torchaudio.list_audio_backends())

class PetDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_sample_rate=22050, num_samples=22050):
        self.data_dir = data_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        # Get list of audio files
        self.audio_files = [f for f in os.listdir(data_dir) if f.endswith('.mp3')]

        # Assuming you have labels - create a simple label system if not
        self.labels = [0 if 'dog' in f else 1 for f in self.audio_files]  # Example label logic

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.audio_files[idx])

        # Load audio file
        signal, sr = librosa.load(audio_path, sr=self.target_sample_rate)

        # Convert to tensor
        signal_tensor = torch.from_numpy(signal).float()

        # Truncate or pad to consistent length
        if len(signal_tensor) > self.num_samples:
            signal_tensor = signal_tensor[:self.num_samples]
        else:
            signal_tensor = torch.nn.functional.pad(signal_tensor, (0, self.num_samples - len(signal_tensor)))

        # Get corresponding label
        label = self.labels[idx]

        return signal_tensor, label


def load_audio(file_path):
    # Using librosa to load audio
    audio, sample_rate = librosa.load(file_path, sr=None)
    return torch.from_numpy(audio), sample_rate
