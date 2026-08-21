import torch

from test_data import test_texts, test_correct_categories_ids
from train import train_model
from vectorizer import vectorizer


model = train_model()

test_X = vectorizer.transform(test_texts)

test_X_tensor = torch.tensor(
    test_X.toarray(),
    dtype=torch.float32,
)

test_y_tensor = torch.tensor(
    test_correct_categories_ids,
    dtype=torch.long,
)

model.eval()

with torch.no_grad():
    output = model(test_X_tensor)
    predictions = output.argmax(dim=1)

accuracy = (
    predictions == test_y_tensor
).float().mean()

print("Predictions:", predictions)
print("Actual:     ", test_y_tensor)
print(f"Test accuracy: {accuracy:.2%}")