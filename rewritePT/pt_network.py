import torch.nn as nn

# 1:1 port of Network class to PyTorch. Everything we did there, from weights to
# forward pass to backward pass is all absorbed in about 20 loc, abstracting
# a lot of what happens under the hood

class DigitNet(nn.Module):
    def __init__(self, num_of_classes = 10):
        super().__init__()

        # activation function
        self.relu = nn.ReLU()

        # linear layers
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_of_classes)

        # flattener
        self.flattener = nn.Flatten()

    def forward(self, x):
        x = self.flattener(x)

        # ReLU + layer in one line
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)

        return out


