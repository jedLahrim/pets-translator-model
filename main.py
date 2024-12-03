from config import MODEL_PATH
from evaluate import evaluate
from train import train


def main():
    # Train the model
    train()

    # Evaluate and generate descriptions
    evaluate(MODEL_PATH)


if __name__ == "__main__":
    main()
