import torch
from torch import nn

from vectorizer import X_tensor

class TextClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)