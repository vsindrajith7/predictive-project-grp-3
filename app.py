import streamlit as st
import joblib
import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

@st.cache_resource
def load_model():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("naive_bayes_model.pkl")
    return tfidf, model

tfidf, model = load_model()

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

st.title("📧 Email Spam Detector")
st.markdown("Paste an email below to check if it is **Spam** or **Ham**.")
st.divider()

col1, col2 = st.columns(2)
if col1.button("Load Spam Example"):
    st.session_state["email"] = "Congratulations! You won a FREE prize worth $1000! Click here NOW to claim. Limited time offer!"
if col2.button("Load Ham Example"):
    st.session_state["email"] = "Hi, just following up on our meeting scheduled for Thursday. Please find the agenda attached."

email_input = st.text_area("Enter Email Content:", value=st.session_state.get("email", ""), height=200)

if st.button("🔍 Classify", type="primary", use_container_width=True):
    if not email_input.strip():
        st.warning("Please enter some email text.")
    else:
        cleaned    = preprocess_text(email_input)
        vec        = tfidf.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba      = model.predict_proba(vec)[0]

        if prediction == 1:
            st.error(f"🚨 SPAM — Confidence: {proba[1]*100:.1f}%")
        else:
            st.success(f"✅ HAM (Not Spam) — Confidence: {proba[0]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Ham", "Spam"], [proba[0], proba[1]], color=["#2ecc71", "#e74c3c"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()
st.subheader("All Model Results")
st.dataframe(pd.DataFrame({
    "Model":     ["Naive Bayes", "SVM", "LSTM"],
    "Accuracy":  ["97.0%", "98.5%", "99.0%"],
    "F1-Score":  ["0.96",  "0.98",  "0.99"],
}).set_index("Model"), use_container_width=True)