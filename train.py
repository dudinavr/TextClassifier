import torch
from torch import nn

from model import TextClassifier
from vectorizer import X_tensor, y_tensor

def train_model():
    model = TextClassifier(
        input_size=X_tensor.shape[1],
        num_classes=5,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    for epoch in range(100):
        output = model(X_tensor)

        loss = criterion(output, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

    return model