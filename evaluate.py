import torch
from torch.utils.data import DataLoader

from config import DATA_DIR, BATCH_SIZE
from dataset import PetDataset
from model import PetPredictorModel  # Import your model class
from preprocess import get_transforms


def evaluate(model_path):
    # Create dataset to determine input length
    dataset = PetDataset(DATA_DIR, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Determine input length
    sample_audio, _ = next(iter(dataloader))
    input_length = sample_audio.shape[-1]

    # Recreate the model with the same architecture
    model = PetPredictorModel(input_length=input_length)

    # Load the saved state dict
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for audio, labels in dataloader:
            # Ensure correct input shape
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() == 2:
                audio = audio.unsqueeze(1)

            outputs = model(audio)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total * 100:.2f}%")