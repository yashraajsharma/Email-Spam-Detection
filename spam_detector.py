"""
Email Spam Detection System
============================
Builds a text classifier from scratch: raw text -> cleaned text ->
numeric features -> trained model -> evaluation.

Instead of picking one model by assumption, this script trains a few
candidates and compares them on the same data, so the final choice is
backed by numbers.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

from dataset import DATA
from preprocessing import preprocess


# --------------------------------------------------------------------------
# STEP 1: Load the data
# --------------------------------------------------------------------------
df = pd.DataFrame(DATA, columns=["text", "label"])
print("Dataset shape:", df.shape)
print(df["label"].value_counts(), "\n")


# --------------------------------------------------------------------------
# STEP 2: Preprocess the text
# --------------------------------------------------------------------------
df["clean_text"] = df["text"].apply(preprocess)


# --------------------------------------------------------------------------
# STEP 3: Train / test split
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")


# --------------------------------------------------------------------------
# STEP 4: Compare feature extraction methods x models
# --------------------------------------------------------------------------
# Two ways to turn text into numbers:
#   - CountVectorizer: raw word counts (Bag-of-Words)
#   - TfidfVectorizer: word counts down-weighted by how common the word is
#     across ALL messages, so generic words matter less than distinctive ones
feature_extractors = {
    "CountVectorizer": CountVectorizer(),
    "TfidfVectorizer": TfidfVectorizer(),
}

candidate_models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
}

results = []
fitted = {}  # keep fitted (vectorizer, model) pairs to reuse the best one later

for fe_name, vectorizer in feature_extractors.items():
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    for model_name, model in candidate_models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="spam")
        rec = recall_score(y_test, y_pred, pos_label="spam")
        f1 = f1_score(y_test, y_pred, pos_label="spam")

        results.append({
            "Features": fe_name, "Model": model_name,
            "Accuracy": round(acc, 3), "Precision": round(prec, 3),
            "Recall": round(rec, 3), "F1": round(f1, 3),
        })
        fitted[(fe_name, model_name)] = (vectorizer, model)

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print("=== Model comparison (sorted by F1-score) ===")
print(results_df.to_string(index=False), "\n")


# --------------------------------------------------------------------------
# STEP 5: Pick the winner and do a full evaluation on it
# --------------------------------------------------------------------------
best_row = results_df.iloc[0]
best_fe, best_model_name = best_row["Features"], best_row["Model"]
vectorizer, model = fitted[(best_fe, best_model_name)]

print(f"Selected model: {best_model_name} + {best_fe} "
      f"(F1={best_row['F1']}, Accuracy={best_row['Accuracy']})\n")

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print("=== Final classification report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name} + {best_fe}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png\n")


# --------------------------------------------------------------------------
# STEP 6: Try it on brand-new messages
# --------------------------------------------------------------------------
new_messages = [
    "Congratulations! You have won a free lottery prize, click here now to claim!",
    "Hey, can we reschedule our meeting to 5pm today?",
    "URGENT: verify your bank account immediately or it will be suspended",
    "Don't forget to bring the notes for tomorrow's class",
]

new_clean = [preprocess(m) for m in new_messages]
new_vec = vectorizer.transform(new_clean)
predictions = model.predict(new_vec)

print("=== Predictions on new messages ===")
for msg, pred in zip(new_messages, predictions):
    print(f"[{pred.upper():5s}]  {msg}")
