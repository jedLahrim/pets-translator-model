import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from config import *
from dataset import PetDataset
from model import PetBehaviorModel


def generate_description(model, audio_features, vocab, max_length=50):
    model.eval()
    with torch.no_grad():
        # Encode audio features
        audio_encoded = model(audio_features)

        # Generate description
        generated_tokens = [vocab['<start>']]
        for _ in range(max_length):
            current_tokens = torch.tensor([generated_tokens]).to(audio_features.device)
            output, _ = model(audio_features, current_tokens)

            # Get the last time step's prediction
            predicted_token = output[0, -1, :].argmax().item()
            generated_tokens.append(predicted_token)

            if predicted_token == vocab['<end>']:
                break

        # Convert tokens back to words
        tokenizer = get_tokenizer('basic_english')
        return ' '.join(vocab.lookup_tokens(generated_tokens[1:-1]))


def evaluate(model_path):
    # Load saved model and vocabulary
    checkpoint = torch.load(model_path)
    vocab = checkpoint['vocab']

    # Recreate model
    model = PetBehaviorModel(
        input_dim=13,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=len(vocab)
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Prepare test dataset
    test_dataset = PetDataset(DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate
    for audio_features, _ in test_loader:
        description = generate_description(model, audio_features, vocab)
        print(f"Generated Description: {description}")


if __name__ == "__main__":
    evaluate(MODEL_PATH)
