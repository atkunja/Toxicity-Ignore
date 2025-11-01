import json
import os
import re

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Step 1: Load Jigsaw dataset ---
df = pd.read_csv("train.csv")
df = df[["comment_text", "toxic"]]
df = df.rename(columns={"comment_text": "text", "toxic": "label"})
df["label"] = df["label"].astype(int)

# --- Step 2: Load feedback (if exists), add words/phrases to feedback_words ---
feedback_words = set()
feedback = None
feedback_path = "cpp_filter/feedback_log.csv"
if os.path.exists(feedback_path):
    feedback = pd.read_csv(feedback_path, names=["text", "label"])
    feedback = feedback.dropna(subset=["text", "label"])
    feedback = feedback[feedback["label"].astype(str).isin(["0", "1"])]
    feedback["label"] = feedback["label"].astype(int)
    # Repeat feedback 100 times for maximum signal!
    feedback = pd.concat([feedback]*100, ignore_index=True)
    print(f"Loaded feedback: {feedback.shape[0]} rows (after repeat)")
    # Add every word, bigram, and full phrase from feedback to feedback_words set
    for phrase in feedback["text"]:
        tokens = re.findall(r'\b\w+\b', str(phrase).lower())
        feedback_words.update(tokens)  # unigrams
        feedback_words.update([' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]) # bigrams
        feedback_words.add(' '.join(tokens))  # full phrase
    # Add feedback rows to main dataframe
    df = pd.concat([df, feedback], ignore_index=True)
else:
    print("No feedback_log.csv found, continuing with Jigsaw data only.")

print(f"Total training rows: {df.shape[0]}")

# --- Step 3: Downsample Jigsaw for feedback effect ---
if feedback is not None:
    feedback_count = feedback.shape[0]
    jigsaw_rows = df.head(len(df) - feedback_count)
    jigsaw_rows = jigsaw_rows.sample(1000, random_state=42)  # Downsample Jigsaw
    feedback_rows = df.tail(feedback_count)
    df = pd.concat([jigsaw_rows, feedback_rows], ignore_index=True)
    print(f"After downsampling: {df.shape[0]} rows (with feedback)")

# --- Step 4: Vectorizer (big vocab, ngrams, include feedback words/phrases) ---
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# --- Step 5: Force feedback words/phrases into vocab ---
orig_vocab = vectorizer.vocabulary_
new_vocab = dict(orig_vocab)
max_idx = max(new_vocab.values(), default=-1)
added = 0
for word in feedback_words:
    if word not in new_vocab:
        max_idx += 1
        new_vocab[word] = max_idx
        added += 1
print(f"Added {added} feedback words/phrases/sentences to vocab.")

# --- Step 6: Re-vectorize with forced vocab ---
vectorizer = TfidfVectorizer(
    vocabulary=new_vocab,
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df["text"])

# --- Step 7: Train logistic regression ---
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X, y)

# --- Step 8: Convert vocab indices to int and export model ---
vocab_clean = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
model_data = {
    "weights": model.coef_[0].tolist(),
    "bias": float(model.intercept_[0]),
    "vocab": vocab_clean
}
with open("model.json", "w") as f:
    json.dump(model_data, f, indent=2)

# --- Step 8b: Persist full sklearn objects for accurate inference ---
dump(
    {
        "vectorizer": vectorizer,
        "classifier": model,
    },
    "model.joblib",
)
print("Saved sklearn vectorizer+classifier to model.joblib")

# --- Step 9: Print final checks for key feedback words ---
for word in ["dyke", "nigger", "nigga", "faggot", "cunt"]:  # Add more if you want to check
    in_vocab = word in vectorizer.vocabulary_
    idx = vectorizer.vocabulary_.get(word)
    w = model.coef_[0][idx] if idx is not None else None
    print(f'"{word}" in vocab? {in_vocab} | Weight: {w}')

print("✅ Model trained on Jigsaw data + feedback (all feedback words/phrases/sentences in vocab) and exported to model.json")
