# 📧 Email Spam Detection System

A machine learning pipeline that classifies emails/SMS messages as **spam**
or **ham (legitimate)**, built from scratch in Python — covering text
preprocessing, feature extraction, model comparison, and evaluation.

Instead of assuming one algorithm, this project **benchmarks 3 models
across 2 feature-extraction techniques** and selects the best performer
based on F1-score, so the final model choice is backed by evidence, not
convention.

---

## Features

- End-to-end pipeline: raw text → cleaned text → numeric features → trained model → evaluation
- Custom text preprocessing (lowercasing, URL stripping, punctuation removal, stopword filtering)
- Compares **Bag-of-Words** vs **TF-IDF** feature extraction
- Compares **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear SVM**
- Automatically selects the best model based on F1-score
- Full evaluation: accuracy, precision, recall, F1-score, confusion matrix
- Tested on unseen, hand-written messages to check real-world generalization
- **Interactive Streamlit web app** — paste any message and get a live spam/ham prediction with confidence score

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** scikit-learn, pandas, matplotlib

---

## Project Structure

```
spam-detector/
├── dataset.py            # Labeled message dataset
├── preprocessing.py       # Shared text-cleaning logic (used by training AND the app)
├── spam_detector.py       # Model comparison + evaluation pipeline
├── train_model.py         # Trains and saves the final chosen model
├── app.py                 # Streamlit web app for live predictions
├── requirements.txt
├── confusion_matrix.png   # Output: confusion matrix of the best model
└── README.md
```

---

## Installation & Usage

```bash
git clone https://github.com/<your-username>/spam-detector.git
cd spam-detector
pip install -r requirements.txt
```

**1. Compare models and see evaluation metrics:**
```bash
python spam_detector.py
```
Prints the model comparison table, the final classification report, saves
`confusion_matrix.png`, and shows predictions on a few new, unseen example
messages.

**2. Train and save the final model** (needed before running the app):
```bash
python train_model.py
```
Saves `model.pkl` and `vectorizer.pkl`.

**3. Launch the interactive web app:**
```bash
streamlit run app.py
```
Opens a browser tab where you can paste any message and get a live
spam/ham prediction with a confidence score.

---

## How It Works

1. **Preprocessing** — text is lowercased, URLs and punctuation are
   stripped, and common stopwords ("the", "is", "and"...) are removed, so
   the model focuses on words that actually carry signal.
2. **Feature extraction** — cleaned text is converted into numeric vectors
   using either:
   - **CountVectorizer** (Bag-of-Words): raw word-frequency counts
   - **TfidfVectorizer**: word counts down-weighted by how common the word
     is across *all* messages, so rare/distinctive words matter more
3. **Model training** — three classifiers are trained on the same
   train/test split: Multinomial Naive Bayes, Logistic Regression, and
   Linear SVM.
4. **Model selection** — all 6 combinations (3 models × 2 feature methods)
   are evaluated on a held-out test set, and the one with the best
   F1-score is selected as the final model.
5. **Evaluation** — accuracy, precision, recall, F1, and a confusion
   matrix quantify how well the final model performs, followed by a
   sanity check on completely new example messages.

---

## Results

Trained and evaluated on a labeled dataset of spam/ham messages,
80/20 train-test split:

| Features        | Model              | Accuracy | Precision | Recall | F1    |
|------------------|--------------------|:--------:|:---------:|:------:|:-----:|
| CountVectorizer  | Logistic Regression| 0.944    | 1.000     | 0.889  | 0.941 |
| CountVectorizer  | Linear SVC         | 0.944    | 1.000     | 0.889  | 0.941 |
| CountVectorizer  | Multinomial NB     | 0.889    | 0.889     | 0.889  | 0.889 |
| TF-IDF           | Multinomial NB     | 0.889    | 0.889     | 0.889  | 0.889 |
| TF-IDF           | Logistic Regression| 0.889    | 0.889     | 0.889  | 0.889 |
| TF-IDF           | Linear SVC         | 0.889    | 0.889     | 0.889  | 0.889 |

**Selected model: Logistic Regression + CountVectorizer**, with **94.4%
accuracy** and **0.941 F1-score** on the test set.

> Note: this repo ships with a small (~90-message) hand-built sample
> dataset for demonstration and reproducibility without external
> downloads. Swapping in a larger public dataset (e.g. the SMS Spam
> Collection on Kaggle, ~5,500+ messages) is a one-line change in
> `dataset.py` and should push accuracy meaningfully higher, since Naive
> Bayes and Logistic Regression both benefit from more training examples.

---

## Why Compare Models Instead of Picking One?

Naive Bayes is the traditional default for spam filtering because it's
fast and simple, and its independence assumption happens to work well
when a few strong "signal words" dominate spam messages. But it isn't
always the most *accurate* choice — here, Logistic Regression and Linear
SVM both outperformed it. Benchmarking multiple models against the same
data is standard ML practice and produces a defensible, evidence-based
model choice rather than one based on convention alone.

---

## Future Improvements

- Train on the full-scale Kaggle SMS Spam Collection dataset
- Add cross-validation instead of a single train/test split for more
  robust comparison
- Try word embeddings (Word2Vec) or a transformer-based model (e.g.
  DistilBERT) as a stronger baseline
- Deploy as a simple web app (Streamlit/Flask) with a live text-input demo
- Add hyperparameter tuning (GridSearchCV) for the winning model

---

## Author

**Yash Raj Sharma**
B.Tech Information Technology, JECRC, Jaipur
[GitHub](https://github.com/yashrajsharma) · [LinkedIn](https://linkedin.com/in/yashrajsharma)
