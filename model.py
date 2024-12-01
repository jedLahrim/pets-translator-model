import torch.nn as nn


class PetPredictorModel(nn.Module):
    def __init__(self, input_length):
        super(PetPredictorModel, self).__init__()

        # Dynamically calculate the output size after convolution and pooling
        def conv_output_length(input_length, kernel_size, stride, padding=0):
            return ((input_length + 2 * padding - kernel_size) // stride) + 1

        # First convolutional layer
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2)
        conv1_out = conv_output_length(input_length, kernel_size=5, stride=2)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)
        pool_out = conv_output_length(conv1_out, kernel_size=2, stride=2)

        # Fully connected layer
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * pool_out, 2)  # Adjust output classes as needed

    def forward(self, x):
        # Ensure input is in the right shape (batch_size, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)
