import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DATA_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS
from dataset import PetDataset
from model import PetPredictorModel
from preprocess import get_transforms


def train():
    dataset = PetDataset(DATA_DIR, transform=get_transforms())
    dataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Determine input length
    sample_audio, sample_labels = next(iter(dataLoader))
    input_length = sample_audio.shape[-1]

    model = PetPredictorModel(input_length=input_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for audio, labels in dataLoader:
            # Ensure correct input shape
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() == 2:
                audio = audio.unsqueeze(1)

            labels = labels.long()

            outputs = model(audio)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

    return model
