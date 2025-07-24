import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load model and vocab
with open("model.json", "r") as f:
    model_data = json.load(f)

vocab = model_data["vocab"]
weights = np.array(model_data["weights"])
bias = model_data["bias"]

# Use same vocab in vectorizer (fit with dummy data so .transform works)
vectorizer = TfidfVectorizer(vocabulary=vocab)
vectorizer.fit(["dummy"])  # Fit with dummy to satisfy scikit-learn's internal check

# Test input
test_sentences = [
    "You are amazing",
    "You're an idiot",
    "Such a terrible idea",
    "I love this!",
    "I hope you die"
]

X_test = vectorizer.transform(test_sentences)

# Manual prediction (dot product + bias)
scores = X_test @ weights + bias
preds = 1 / (1 + np.exp(-scores))  # sigmoid

for i, sentence in enumerate(test_sentences):
    label = "❌ Toxic" if preds[i] > 0.5 else "✅ Safe"
    print(f"{sentence} → {label} ({preds[i]:.2f})")
