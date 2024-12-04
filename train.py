import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from config import *
from dataset import PetDataset
from model import PetBehaviorModel


def tokenize_captions(captions):
    tokenizer = get_tokenizer('basic_english')
    return [tokenizer(caption) for caption in captions]


def build_vocabulary(captions):
    tokenized_captions = tokenize_captions(captions)
    vocab = build_vocab_from_iterator(tokenized_captions, specials=['<unk>', '<pad>', '<start>', '<end>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def train():
    # Prepare dataset
    labels_dict = {
        'dog_46.wav': 'the dog is actually sick',
        'dog-barking-101722.mp3': 'your dog want to play',
        'dog-barking-101723.wav': 'this dog is smelling a danger',
        # Add more mappings
    }
    dataset = PetDataset(DATA_DIR, labels_dict)

    # Build vocabulary
    captions = [caption for _, caption in dataset.audio_files]
    vocab = build_vocabulary(captions)
    vocab_size = len(vocab)

    # Prepare dataloader
    def collate_fn(batch):
        audio_features, captions = zip(*batch)

        # Tokenize and convert captions to tensor
        tokenized_captions = [torch.tensor([vocab[token] for token in tokenize_captions([caption])[0]]) for caption in
                              captions]
        padded_captions = pad_sequence(tokenized_captions, batch_first=True)

        return torch.stack(audio_features), padded_captions

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = PetBehaviorModel(
        input_dim=13,  # MFCC features
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=vocab_size
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for audio_features, captions in dataloader:
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(audio_features, captions[:, :-1])

            # Compute loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    # Save model and vocabulary
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, MODEL_PATH)


if __name__ == "__main__":
    train()
