from joblib import load

from toxicity_heuristics import contains_high_severity_toxicity

# Load trained sklearn objects
bundle = load("model.joblib")
pipeline = bundle.get("pipeline")

test_sentences = [
    "You are amazing",
    "You're an idiot",
    "Such a terrible idea",
    "I love this!",
    "I hope you die",
]

if pipeline is not None:
    preds = pipeline.predict_proba(test_sentences)[:, 1]
else:
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    preds = classifier.predict_proba(vectorizer.transform(test_sentences))[:, 1]

for sentence, prob in zip(test_sentences, preds):
    label = "❌ Toxic" if prob > 0.5 else "✅ Safe"
    print(f"{sentence} → {label} ({prob:.2f})")

heuristic_sentence = "hi ashmit u faggot"
print(
    f"Heuristic check for '{heuristic_sentence}':",
    "❌ Toxic" if contains_high_severity_toxicity(heuristic_sentence) else "✅ Safe",
)
