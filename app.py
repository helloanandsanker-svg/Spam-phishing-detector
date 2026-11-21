from pathlib import Path
import joblib
import streamlit as st

from src.preprocess import clean_text

ROOT_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.joblib"
MODEL_PATH = ROOT_DIR / "models" / "spam_model.joblib"

st.set_page_config(page_title="SMS & Email Spam / Phishing Detector", page_icon="🛡️")


@st.cache_resource
def load_artifacts():
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        return None, None
    return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)


st.title("SMS & Email Spam / Phishing Detector")
st.write("Paste an SMS or email message below to check for spam or phishing patterns using TF-IDF + Logistic Regression.")

vectorizer, model = load_artifacts()
if vectorizer is None or model is None:
    st.error("Trained model files not found. Run `python src/train.py` first to generate `models/`.")
    st.stop()

sample_messages = {
    "Select a sample message...": "",
    "Legitimate SMS": "Hey, are we still meeting at the library around 5 PM today?",
    "Phishing / Scam SMS": "URGENT: Your bank account will be suspended today. Verify KYC now at http://bit.ly/fake-kyc-update",
    "Phishing Email": "Security Alert: Unusual sign-in detected from a new device. Reset your password immediately via our portal.",
}

choice = st.selectbox("Try a sample:", list(sample_messages.keys()))
default_text = sample_messages[choice]

message = st.text_area("Message text", value=default_text, height=140, placeholder="Type or paste an SMS or email...")

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        features = vectorizer.transform([cleaned])
        proba = model.predict_proba(features)[0]
        risk_score = float(proba[1])

        st.divider()
        if risk_score >= 0.5:
            st.error(f"High Risk: Likely Spam / Phishing ({risk_score:.1%} confidence)")
        else:
            st.success(f"Low Risk: Likely Legitimate / Ham ({1 - risk_score:.1%} confidence)")

        with st.expander("See preprocessed text"):
            st.code(cleaned or "(empty after cleaning)", language="text")