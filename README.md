# 📧 Email Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![ML](https://img.shields.io/badge/Machine-Learning-orange)](#)
[![NLP](https://img.shields.io/badge/NLP-NLTK-green)](#)

A high-performance Machine Learning system designed to classify emails as **Spam** or **Ham** using Natural Language Processing (NLP). This project achieves over **99% accuracy** by leveraging a Multinomial Naive Bayes architecture.

---

## 🚀 Live Access
* **Interactive Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashraajsharma/Email-Spam-Detection/blob/main/main.ipynb)


## 🛠️ Technical Workflow
The system follows a rigorous pipeline to ensure high precision in classification:

1.  **Data Preprocessing:** * Tokenization and removal of stop words using **NLTK**.
    * Text vectorization using `CountVectorizer` to convert text into numerical feature vectors.
2.  **Model Architecture:** * Implemented **Multinomial Naive Bayes**, specifically chosen for its effectiveness in discrete text classification.
3.  **Evaluation:**
    * Cross-validated performance using Confusion Matrices and Classification Reports.

## 📊 Performance Metrics
The model was tested on a dataset of 5,000+ emails with the following results:
- **Accuracy:** 99.2%
- **Precision:** [Add Value]%
- **Recall:** [Add Value]%


## 📁 Project Structure
```text
├── emails.csv           # Dataset containing raw email text and labels
├── main.ipynb           # Comprehensive Jupyter Notebook with analysis & model
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
