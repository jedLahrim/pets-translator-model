import torch.nn as nn


class PetBehaviorModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, vocab_size):
        super(PetBehaviorModel, self).__init__()

        # Audio feature extraction
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Text generation components
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, audio_features, captions=None):
        # Encode audio features
        audio_encoded = self.audio_encoder(audio_features.transpose(1, 2))
        audio_encoded = audio_encoded.mean(dim=2)  # Global average pooling

        # Text generation (if captions provided)
        if captions is not None:
            embedded_captions = self.embedding(captions)
            lstm_out, _ = self.lstm(embedded_captions)
            outputs = self.fc(lstm_out)
            return outputs, audio_encoded

        return audio_encoded
