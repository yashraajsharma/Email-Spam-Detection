"""
Trains the final model (Logistic Regression + CountVectorizer — the
winner from the comparison in spam_detector.py) on the FULL dataset
(no train/test split, since this is the model we're shipping) and saves
it to disk so app.py can load it instantly without retraining.

Run this once before starting the Streamlit app:
    python train_model.py
"""

import pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from dataset import DATA
from preprocessing import preprocess

df = pd.DataFrame(DATA, columns=["text", "label"])
df["clean_text"] = df["text"].apply(preprocess)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved model.pkl and vectorizer.pkl")
print(f"Trained on {len(df)} messages, vocabulary size: {len(vectorizer.vocabulary_)}")
