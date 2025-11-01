import json
import os
import re

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

DATA_PATH = "train.csv"
FEEDBACK_PATH = "cpp_filter/feedback_log.csv"
REPEAT_FEEDBACK = 100
MAX_JIGSAW_ROWS = 120000

# --- Step 1: Load dataset ---
df = pd.read_csv(DATA_PATH)
df = df[["comment_text", "toxic"]]
df = df.rename(columns={"comment_text": "text", "toxic": "label"})
df["text"] = df["text"].fillna("").astype(str)
df["label"] = df["label"].astype(int)

# --- Step 2: Load feedback (if exists) and enrich vocabulary ---
feedback_words = set()
if os.path.exists(FEEDBACK_PATH):
    feedback = pd.read_csv(FEEDBACK_PATH, names=["text", "label"])
    feedback = feedback.dropna(subset=["text", "label"])
    feedback = feedback[feedback["label"].astype(str).isin(["0", "1"])]
    feedback["label"] = feedback["label"].astype(int)
    feedback["text"] = feedback["text"].astype(str)
    feedback = pd.concat([feedback] * REPEAT_FEEDBACK, ignore_index=True)
    print(f"Loaded feedback: {feedback.shape[0]} rows (after repeat)")
    for phrase in feedback["text"]:
        tokens = re.findall(r"\b\w+\b", phrase.lower())
        feedback_words.update(tokens)
        feedback_words.update(
            [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
        )
        feedback_words.add(" ".join(tokens))
    df = pd.concat([df, feedback], ignore_index=True)
else:
    print("No feedback_log.csv found, continuing with Jigsaw data only.")

print(f"Total rows after feedback merge: {df.shape[0]}")

# --- Step 3: Optional downsampling for speed ---
if len(df) > MAX_JIGSAW_ROWS:
    toxic_rows = df[df["label"] == 1]
    remaining = max(MAX_JIGSAW_ROWS - len(toxic_rows), 0)
    non_toxic_rows = df[df["label"] == 0].sample(remaining, random_state=42)
    df = pd.concat([toxic_rows, non_toxic_rows], ignore_index=True)
    print(f"Downsampled dataset to {df.shape[0]} rows.")
else:
    print(f"Using full dataset of {df.shape[0]} rows.")

texts = df["text"]
labels = df["label"]

# --- Step 4: Advanced TF-IDF + char model for main API ---
word_tfidf = TfidfVectorizer(
    max_features=60000,
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_features=30000,
    sublinear_tf=True,
)
feature_union = FeatureUnion(
    [
        ("word", word_tfidf),
        ("char", char_tfidf),
    ]
)
advanced_pipeline = Pipeline(
    [
        ("features", feature_union),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="saga",
                penalty="l2",
                C=1.0,
            ),
        ),
    ]
)

X_train, X_valid, y_train, y_valid = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)
advanced_pipeline.fit(X_train, y_train)
val_preds = advanced_pipeline.predict(X_valid)
val_probs = advanced_pipeline.predict_proba(X_valid)[:, 1]
print("Validation metrics (advanced pipeline):")
print(classification_report(y_valid, val_preds, digits=3))
print(f"Validation ROC-AUC: {roc_auc_score(y_valid, val_probs):.4f}")

advanced_pipeline.fit(texts, labels)
dump({"pipeline": advanced_pipeline}, "model.joblib")
print("Saved advanced sklearn pipeline to model.joblib")

# --- Step 5: Legacy bag-of-words logistic model for C++/fallback ---
count_vectorizer = CountVectorizer(
    max_features=60000,
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
)
count_vectorizer.fit(texts)
legacy_vocab = dict(count_vectorizer.vocabulary_)
max_idx = max(legacy_vocab.values(), default=-1)
added_tokens = 0
for token in feedback_words:
    if token not in legacy_vocab:
        max_idx += 1
        legacy_vocab[token] = max_idx
        added_tokens += 1
if added_tokens:
    print(f"Added {added_tokens} feedback tokens to legacy vocabulary.")

count_vectorizer = CountVectorizer(
    vocabulary=legacy_vocab,
    lowercase=True,
    ngram_range=(1, 2),
)
count_vectorizer.fit(texts)
X_legacy = count_vectorizer.transform(texts)
legacy_model = LogisticRegression(max_iter=1000, class_weight="balanced")
legacy_model.fit(X_legacy, labels)

legacy_vocab_clean = {k: int(v) for k, v in count_vectorizer.vocabulary_.items()}
legacy_model_data = {
    "weights": legacy_model.coef_[0].tolist(),
    "bias": float(legacy_model.intercept_[0]),
    "vocab": legacy_vocab_clean,
}
with open("model.json", "w") as f:
    json.dump(legacy_model_data, f, indent=2)
print("Exported legacy bag-of-words model to model.json (for C++ fallback)")

key_terms = {"dyke", "nigger", "nigga", "faggot", "cunt", "bitch", "motherfucker"}
for term in sorted(key_terms):
    in_vocab = term in count_vectorizer.vocabulary_
    idx = count_vectorizer.vocabulary_.get(term)
    weight = legacy_model.coef_[0][idx] if idx is not None else None
    print(f'"{term}" in vocab? {in_vocab} | Weight: {weight}')

print("✅ Training complete.")
