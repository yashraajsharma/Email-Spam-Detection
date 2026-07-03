"""
Streamlit app for the Email Spam Detection System.

Run with:
    streamlit run app.py

Make sure you've run `python train_model.py` first so model.pkl and
vectorizer.pkl exist.
"""

import pickle
import streamlit as st

from preprocessing import preprocess

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


model, vectorizer = load_model()

st.title("📧 Email Spam Detector")
st.write(
    "Paste an email or message below and the model will predict whether "
    "it's **spam** or **legitimate (ham)**, trained using Logistic "
    "Regression on Bag-of-Words features."
)

example_messages = {
    "-- Choose an example --": "",
    "Spam example": "Congratulations! You've won a $1000 gift card. Click here now to claim your prize before it expires!",
    "Ham example": "Hey, are we still meeting for lunch tomorrow at 1pm? Let me know if that works.",
}

choice = st.selectbox("Try an example, or type your own below:", list(example_messages.keys()))
default_text = example_messages[choice]

user_input = st.text_area("Message text", value=default_text, height=150,
                           placeholder="Paste an email or SMS message here...")

if st.button("Check message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = preprocess(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        spam_prob = proba[list(model.classes_).index("spam")]

        if prediction == "spam":
            st.error(f"🚨 This looks like **SPAM** (confidence: {spam_prob:.1%})")
        else:
            st.success(f"✅ This looks like **HAM** — legitimate message (spam confidence: {spam_prob:.1%})")

        with st.expander("See how the model processed this message"):
            st.write("**Original text:**")
            st.code(user_input)
            st.write("**After preprocessing (lowercased, punctuation/stopwords removed):**")
            st.code(cleaned if cleaned else "(empty after cleaning)")
            st.write(f"**Predicted class:** `{prediction}`")
            st.write(f"**Spam probability:** {spam_prob:.3f}  |  **Ham probability:** {1 - spam_prob:.3f}")

st.divider()
st.caption(
    "Model: Logistic Regression + CountVectorizer, selected after comparing "
    "3 models × 2 feature-extraction methods on held-out test data. "
    "Built by Yash Raj Sharma."
)
