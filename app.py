import json
import os
import subprocess

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load

from toxicity_heuristics import heuristic_scores

MODEL_ARTIFACT = "model.joblib"
LEGACY_MODEL_FILE = "model.json"
OVERRIDES_FILE = "cpp_filter/instant_overrides.json"
FEEDBACK_LOG = "cpp_filter/feedback_log.csv"

class ToxicityModel:
    """Wraps model loading and inference so Flask uses consistent ML features."""

    def __init__(self):
        self.mode = None
        self.pipeline = None
        self.vectorizer = None
        self.classifier = None
        self.vocab = None
        self.weights = None
        self.bias = None
        self.label_names: list[str] = ["toxic"]
        self._load()

    def _load(self):
        # Prefer the full sklearn artifact for accurate predictions.
        if os.path.exists(MODEL_ARTIFACT):
            try:
                bundle = load(MODEL_ARTIFACT)
                if isinstance(bundle, dict) and "pipeline" in bundle:
                    self.pipeline = bundle["pipeline"]
                    self.label_names = bundle.get("labels", ["toxic"])
                    self.vectorizer = None
                    self.classifier = None
                    self.vocab = None
                    self.weights = None
                    self.bias = None
                    self.mode = "pipeline"
                    feature_summary = []
                    for step_name, estimator in getattr(self.pipeline, "steps", []):
                        if step_name == "features" and hasattr(
                            estimator, "transformer_list"
                        ):
                            for name, transformer in estimator.transformer_list:
                                vocab_size = len(getattr(transformer, "vocabulary_", {}))
                                feature_summary.append(f"{name}:{vocab_size}")
                            break
                    summary = ", ".join(feature_summary) if feature_summary else "pipeline loaded"
                    print(f"Loaded sklearn pipeline from model.joblib ({summary})")
                    return
                if isinstance(bundle, dict) and "vectorizer" in bundle and "classifier" in bundle:
                    self.pipeline = None
                    self.vectorizer = bundle["vectorizer"]
                    self.classifier = bundle["classifier"]
                    self.vocab = None
                    self.weights = None
                    self.bias = None
                    self.label_names = bundle.get("labels", ["toxic"])
                    self.mode = "sklearn"
                    vocab_size = len(getattr(self.vectorizer, "vocabulary_", {}))
                    print(f"Loaded sklearn model.joblib (vocab size: {vocab_size})")
                    return
                raise ValueError("Unexpected model.joblib structure.")
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Failed to load joblib model: {exc}. Falling back to legacy JSON.")

        # Fall back to legacy behaviour if joblib artifact missing.
        if os.path.exists(LEGACY_MODEL_FILE):
            with open(LEGACY_MODEL_FILE) as f:
                model_data = json.load(f)
            self.vocab = model_data["vocab"]
            self.weights = np.array(model_data["weights"])
            self.bias = model_data["bias"]
            self.pipeline = None
            self.label_names = ["toxic"]
            self.mode = "legacy"
            print(f"Loaded legacy model.json (vocab size: {len(self.vocab)})")
            return

        raise FileNotFoundError(
            "No trained model artifacts found. Run train.py to create model.joblib."
        )

    def reload(self):
        """Reload model after retraining."""
        self._load()

    def empty_scores(self) -> dict[str, float]:
        """Return a zero-filled score dictionary for all known labels."""
        return {label: 0.0 for label in self.label_names}

    def predict_scores(self, text: str) -> dict[str, float]:
        """Return a dict of label -> probability."""
        scores = self.empty_scores()
        if not text:
            return scores

        if self.mode == "pipeline":
            probs = self.pipeline.predict_proba([text])
            if isinstance(probs, list):
                # Some sklearn versions return a list of per-class outputs.
                probs = np.vstack([p[:, 1] if p.ndim > 1 else p for p in probs]).T
            for label, prob in zip(self.label_names, probs[0]):
                scores[label] = float(prob)
            return scores

        if self.mode == "sklearn":
            features = self.vectorizer.transform([text])
            prob = self.classifier.predict_proba(features)[0, 1]
            scores["toxic"] = float(prob)
            return scores

        if self.mode == "legacy":
            vec = self._vectorize_legacy(text)
            prob = sigmoid(np.dot(self.weights, vec) + self.bias)
            scores["toxic"] = float(prob)
            return scores

        raise RuntimeError("Model not loaded.")

    def predict_proba(self, text: str) -> float:
        """Return legacy single toxicity probability (backwards compatibility)."""
        return self.predict_scores(text).get("toxic", 0.0)

    def _vectorize_legacy(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vec = np.zeros(len(self.vocab))
        for word in tokens:
            idx = self.vocab.get(word)
            if idx is not None:
                vec[idx] += 1
        return vec


toxicity_model = ToxicityModel()

# --- Overrides Handling ---

def load_overrides():
    if os.path.exists(OVERRIDES_FILE):
        with open(OVERRIDES_FILE) as f:
            return json.load(f)
    return {}

def save_override(text, label):
    overrides = load_overrides()
    overrides[text.strip().lower()] = int(label)
    with open(OVERRIDES_FILE, "w") as f:
        json.dump(overrides, f)

# --- Feedback Logging ---

def log_feedback(text, label, repeat=100):
    with open(FEEDBACK_LOG, "a") as f:
        for _ in range(repeat):
            f.write(f'"{text}",{label}\n')

def evaluate_text(text: str, overrides: dict[str, int]):
    """Return (label, prob, scores dict, source) for a given text."""
    norm_text = text.strip().lower()
    scores = toxicity_model.empty_scores()
    source = "model"

    if norm_text in overrides:
        override_label = int(overrides[norm_text])
        scores["toxic"] = float(override_label)
        label = "toxic" if override_label else "safe"
        prob = float(override_label)
        source = "override"
        return label, prob, scores, source

    heuristics = heuristic_scores(text)
    if heuristics:
        for label_name, value in heuristics.items():
            if label_name in scores:
                scores[label_name] = max(scores[label_name], value)
            else:
                scores[label_name] = value
        scores["toxic"] = max(scores.get("toxic", 0.0), heuristics.get("toxic", 1.0))
        label = "toxic"
        prob = scores["toxic"]
        source = "heuristic"
        return label, prob, scores, source

    scores = toxicity_model.predict_scores(text)
    prob = scores.get("toxic", 0.0)
    label = "toxic" if prob > 0.5 else "safe"
    return label, prob, scores, source

# --- Auto Retrain after feedback ---

def retrain_model():
    try:
        subprocess.run(["python3", "train.py"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Retraining failed: {exc}")
        raise
    toxicity_model.reload()
    print("Reloaded model artifacts!")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

app = Flask(__name__)
CORS(app)  # Allow frontend to access API

# --- WEB FORM (for manual testing) ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    score = None
    text = ""
    feedback_msg = ""
    scores = None
    source = ""
    overrides = load_overrides()
    if request.method == "POST":
        text = request.form.get("text", "")
        norm_text = text.strip().lower()
        if "feedback" in request.form:
            label = int(request.form["feedback"])
            save_override(norm_text, label)
            log_feedback(text, label)
            try:
                retrain_model()
                feedback_msg = f"Thank you! '{text}' is now marked as {'Toxic' if label else 'Safe'} and model is updated."
            except subprocess.CalledProcessError:
                feedback_msg = (
                    f"Feedback saved for '{text}', but model retraining failed. Check server logs."
                )
        else:
            label, prob, scores, source = evaluate_text(text, overrides)
            prediction = "❌ Toxic" if label == "toxic" else "✅ Safe"
            if source == "override":
                score = "override"
            elif source == "heuristic":
                score = "heuristic"
            else:
                score = f"{prob:.3f}"
    score_list = ""
    if scores:
        ordered_labels = list(toxicity_model.label_names)
        extra_labels = [
            label for label in scores.keys() if label not in toxicity_model.label_names
        ]
        for label in sorted(extra_labels):
            ordered_labels.append(label)
        items = "".join(
            f"<li>{label}: {scores[label]:.3f}</li>" for label in ordered_labels if label in scores
        )
        score_list = f"<ul>{items}</ul>"
    return f"""
    <html>
    <head><title>Simple Local Toxicity Filter</title></head>
    <body style="font-family:sans-serif; margin:40px;">
      <h2>Local AI Toxicity Filter</h2>
      <form method="POST">
        <input name="text" size="60" value="{text}" autofocus>
        <input type="submit" value="Check">
      </form>
      {'<h3>Result: ' + prediction + ' (' + score + ')</h3>' if prediction else ''}
      {f"<p><em>Source: {source}</em></p>" if prediction else ''}
      {score_list if prediction else ''}
      {f'''
        <form method="POST">
            <input type="hidden" name="text" value="{text}">
            <button name="feedback" value="1">Mark as Toxic</button>
            <button name="feedback" value="0">Mark as Safe</button>
        </form>
      ''' if prediction else ''}
      {f"<p style='color:green'>{feedback_msg}</p>" if feedback_msg else ''}
    </body>
    </html>
    """

# --- API ENDPOINTS (for React/Next.js frontend) ---
@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json()
    text = data.get("text", "")
    overrides = load_overrides()
    label, prob, scores, source = evaluate_text(text, overrides)
    return jsonify({"label": label, "prob": prob, "scores": scores, "source": source})

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json()
    text = data.get("text", "")
    label = int(data.get("label", 0))
    norm_text = text.strip().lower()
    save_override(norm_text, label)
    log_feedback(text, label)
    try:
        retrain_model()
    except subprocess.CalledProcessError:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Feedback saved, but model retraining failed. Check server logs.",
                }
            ),
            500,
        )
    return jsonify({"success": True, "message": f"Feedback received for '{text}'."})

if __name__ == "__main__":
    app.run(debug=True)
