import numpy as np
from joblib import load

from toxicity_heuristics import heuristic_scores

# Load trained sklearn objects
bundle = load("model.joblib")
pipeline = bundle.get("pipeline")
label_names = bundle.get("labels", ["toxic"])

test_sentences = [
    "You are amazing",
    "You're an idiot",
    "Such a terrible idea",
    "I love this!",
    "I hope you die",
]

if pipeline is not None:
    probs = pipeline.predict_proba(test_sentences)
    if isinstance(probs, list):
        # Some versions may return a list of per-class arrays.
        probs = np.vstack([p[:, 1] if p.ndim > 1 else p for p in probs]).T
    preds = probs
else:
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    probs = classifier.predict_proba(vectorizer.transform(test_sentences))[:, 1]
    preds = probs.reshape(-1, 1)

for sentence, prob_vector in zip(test_sentences, preds):
    score_line = ", ".join(
        f"{label}:{prob:.2f}" for label, prob in zip(label_names, prob_vector)
    )
    print(f"{sentence} → [{score_line}]")

heuristic_sentence = "hi ashmit u faggot"
print(f"Heuristic check for '{heuristic_sentence}': {heuristic_scores(heuristic_sentence)}")
