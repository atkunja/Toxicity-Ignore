from joblib import load

from toxicity_heuristics import contains_high_severity_toxicity

# Load trained sklearn objects
bundle = load("model.joblib")
vectorizer = bundle["vectorizer"]
classifier = bundle["classifier"]

# Test input
test_sentences = [
    "You are amazing",
    "You're an idiot",
    "Such a terrible idea",
    "I love this!",
    "I hope you die"
]

X_test = vectorizer.transform(test_sentences)
preds = classifier.predict_proba(X_test)[:, 1]

for i, sentence in enumerate(test_sentences):
    label = "❌ Toxic" if preds[i] > 0.5 else "✅ Safe"
    print(f"{sentence} → {label} ({preds[i]:.2f})")

heuristic_sentence = "hi ashmit u faggot"
print(
    f"Heuristic check for '{heuristic_sentence}':",
    "❌ Toxic" if contains_high_severity_toxicity(heuristic_sentence) else "✅ Safe",
)
