from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from train_data import train_texts, train_correct_categories_ids

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(train_texts)

X_tensor = torch.tensor(
    X.toarray(),
    dtype=torch.float32,
)

y_tensor = torch.tensor(
    train_correct_categories_ids,
    dtype=torch.long,
)