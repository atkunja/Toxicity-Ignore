import json
import os
import subprocess

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load

MODEL_ARTIFACT = "model.joblib"
LEGACY_MODEL_FILE = "model.json"
OVERRIDES_FILE = "cpp_filter/instant_overrides.json"
FEEDBACK_LOG = "cpp_filter/feedback_log.csv"

class ToxicityModel:
    """Wraps model loading and inference so Flask uses consistent ML features."""

    def __init__(self):
        self.mode = None
        self.vectorizer = None
        self.classifier = None
        self.vocab = None
        self.weights = None
        self.bias = None
        self._load()

    def _load(self):
        # Prefer the full sklearn artifact for accurate predictions.
        if os.path.exists(MODEL_ARTIFACT):
            try:
                bundle = load(MODEL_ARTIFACT)
                self.vectorizer = bundle["vectorizer"]
                self.classifier = bundle["classifier"]
                self.mode = "sklearn"
                vocab_size = len(getattr(self.vectorizer, "vocabulary_", {}))
                print(f"Loaded sklearn model.joblib (vocab size: {vocab_size})")
                return
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Failed to load joblib model: {exc}. Falling back to legacy JSON.")

        # Fall back to legacy behaviour if joblib artifact missing.
        if os.path.exists(LEGACY_MODEL_FILE):
            with open(LEGACY_MODEL_FILE) as f:
                model_data = json.load(f)
            self.vocab = model_data["vocab"]
            self.weights = np.array(model_data["weights"])
            self.bias = model_data["bias"]
            self.mode = "legacy"
            print(f"Loaded legacy model.json (vocab size: {len(self.vocab)})")
            return

        raise FileNotFoundError(
            "No trained model artifacts found. Run train.py to create model.joblib."
        )

    def reload(self):
        """Reload model after retraining."""
        self._load()

    def predict_proba(self, text: str) -> float:
        """Return toxicity probability for raw text."""
        if not text:
            return 0.0

        if self.mode == "sklearn":
            features = self.vectorizer.transform([text])
            prob = self.classifier.predict_proba(features)[0, 1]
            return float(prob)

        if self.mode == "legacy":
            vec = self._vectorize_legacy(text)
            prob = sigmoid(np.dot(self.weights, vec) + self.bias)
            return float(prob)

        raise RuntimeError("Model not loaded.")

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
            if norm_text in overrides:
                prediction = "❌ Toxic" if overrides[norm_text] else "✅ Safe"
                score = "override"
            else:
                prob = toxicity_model.predict_proba(text)
                prediction = "❌ Toxic" if prob > 0.5 else "✅ Safe"
                score = f"{prob:.3f}"
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
    norm_text = text.strip().lower()
    overrides = load_overrides()
    if norm_text in overrides:
        label = "toxic" if overrides[norm_text] else "safe"
        prob = 1.0 if overrides[norm_text] else 0.0
    else:
        prob = toxicity_model.predict_proba(text)
        label = "toxic" if prob > 0.5 else "safe"
    return jsonify({"label": label, "prob": prob})

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
