import json
import numpy as np
import os
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_FILE = "model.json"
OVERRIDES_FILE = "cpp_filter/instant_overrides.json"
FEEDBACK_LOG = "cpp_filter/feedback_log.csv"

# --- Model Loading ---

def load_model():
    with open(MODEL_FILE) as f:
        model = json.load(f)
    VOCAB = model["vocab"]
    WEIGHTS = np.array(model["weights"])
    BIAS = model["bias"]
    print(f'Loaded vocab size: {len(VOCAB)}')
    return model, VOCAB, WEIGHTS, BIAS

model, VOCAB, WEIGHTS, BIAS = load_model()

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
    subprocess.run(["python3", "train.py"])
    global model, VOCAB, WEIGHTS, BIAS
    model, VOCAB, WEIGHTS, BIAS = load_model()
    print("Reloaded model.json!")
    print(f'Does "dyke" exist? {"dyke" in VOCAB}')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def vectorize(text):
    tokens = text.lower().split()
    vec = np.zeros(len(VOCAB))
    for word in tokens:
        idx = VOCAB.get(word)
        if idx is not None:
            vec[idx] += 1
    return vec

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
            retrain_model()
            feedback_msg = f"Thank you! '{text}' is now marked as {'Toxic' if label else 'Safe'} and model is updated."
        else:
            if norm_text in overrides:
                prediction = "❌ Toxic" if overrides[norm_text] else "✅ Safe"
                score = "override"
            else:
                vec = vectorize(text)
                prob = sigmoid(np.dot(WEIGHTS, vec) + BIAS)
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
        vec = vectorize(text)
        prob = float(sigmoid(np.dot(WEIGHTS, vec) + BIAS))
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
    retrain_model()
    return jsonify({"success": True, "message": f"Feedback received for '{text}'."})

if __name__ == "__main__":
    app.run(debug=True)
