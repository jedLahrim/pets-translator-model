from evaluate import evaluate
from train import train

if __name__ == "__main__":
    train()
    evaluate("model.pth")
