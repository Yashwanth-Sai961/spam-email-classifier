"""
Streamlit UI for the Spam Email Classifier.

This module ONLY handles presentation (UI/UX). The underlying machine
learning pipeline (TF-IDF vectorizer + trained classifier) and the text
preprocessing logic are untouched and are loaded/imported exactly as
they exist in the project.

Project structure assumptions (unchanged):
    models/spam_classifier.joblib
    models/tfidf_vectorizer.joblib
    src/preprocess.py -> preprocess_text()
"""

import os
import sys
import time

import joblib
import streamlit as st

# --------------------------------------------------------------------------
# Path setup
# --------------------------------------------------------------------------
# BASE_DIR points to the project root (one level above this file's folder,
# i.e. .../spam_email_classifier)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_classifier.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

# Ensure the project root is importable so `from src.preprocess import
# preprocess_text` works regardless of which directory streamlit was
# launched from.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import preprocess_text  # noqa: E402  (import after sys.path setup)


# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------
# Resource loading (cached so the model/vectorizer load only once)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained spam classifier from disk.

    Returns:
        The deserialized scikit-learn classifier object.

    Raises:
        FileNotFoundError: If the model file does not exist at MODEL_PATH.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_vectorizer():
    """Load the fitted TF-IDF vectorizer from disk.

    Returns:
        The deserialized scikit-learn TfidfVectorizer object.

    Raises:
        FileNotFoundError: If the vectorizer file does not exist at
            VECTORIZER_PATH.
    """
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer file not found at: {VECTORIZER_PATH}")
    return joblib.load(VECTORIZER_PATH)


# --------------------------------------------------------------------------
# Prediction logic
# --------------------------------------------------------------------------
def predict_message(raw_text, model, vectorizer):
    """Run the full prediction pipeline on a raw text message.

    Applies the project's existing preprocessing function before
    vectorizing, then returns the predicted label along with class
    probabilities when available.

    Args:
        raw_text: The raw, unprocessed user-submitted text.
        model: The trained classifier.
        vectorizer: The fitted TF-IDF vectorizer.

    Returns:
        A tuple of (prediction, spam_probability, ham_probability).
        Probabilities are None if the model does not support
        predict_proba.
    """
    cleaned = preprocess_text(raw_text)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]

    spam_prob = None
    ham_prob = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_labels = list(model.classes_)
        if "spam" in class_labels:
            spam_prob = probabilities[class_labels.index("spam")]
        if "ham" in class_labels:
            ham_prob = probabilities[class_labels.index("ham")]

    return prediction, spam_prob, ham_prob


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_custom_css():
    """Inject the premium dark glassmorphism theme into the page."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', 'Poppins', sans-serif;
            }

            .stApp {
                background: radial-gradient(circle at 20% 20%, #1f1147 0%, #0d0b21 45%, #05040f 100%);
                background-attachment: fixed;
            }

            #MainMenu, header, footer {visibility: hidden;}

            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 3rem;
                max-width: 780px;
            }

            /* ---------- Header ---------- */
            .app-header {
                text-align: center;
                padding: 1.5rem 1rem 0.5rem 1rem;
                animation: fadeInDown 0.8s ease-out;
            }

            .app-title {
                font-family: 'Poppins', sans-serif;
                font-weight: 800;
                font-size: 2.6rem;
                background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
                background-size: 200% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: shimmer 6s linear infinite;
                margin-bottom: 0.2rem;
            }

            .app-subtitle {
                color: #b6b0d8;
                font-size: 1.05rem;
                font-weight: 400;
                letter-spacing: 0.3px;
            }

            @keyframes shimmer {
                0% { background-position: 0% center; }
                100% { background-position: 200% center; }
            }

            @keyframes fadeInDown {
                from { opacity: 0; transform: translateY(-16px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            /* ---------- Glass card ---------- */
            .glass-card {
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.12);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
                border-radius: 22px;
                padding: 1.8rem 1.8rem 1.4rem 1.8rem;
                margin-top: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
                animation: fadeIn 0.6s ease-out;
            }

            .glass-card h3 {
                color: #e7e4ff;
                font-family: 'Poppins', sans-serif;
                font-weight: 600;
                margin-bottom: 0.9rem;
                font-size: 1.15rem;
            }

            /* ---------- Text area ---------- */
            .stTextArea textarea {
                background: rgba(0, 0, 0, 0.25) !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-radius: 14px !important;
                color: #f1f0ff !important;
                font-size: 1rem !important;
                padding: 1rem !important;
                transition: border 0.3s ease, box-shadow 0.3s ease;
            }

            .stTextArea textarea:focus {
                border: 1px solid #a78bfa !important;
                box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.25) !important;
            }

            .stTextArea textarea::placeholder {
                color: #8b85b3 !important;
            }

            /* ---------- Predict button ---------- */
            div.stButton > button {
                width: 100%;
                background: linear-gradient(90deg, #7c3aed, #2563eb);
                color: white;
                font-weight: 600;
                font-size: 1.05rem;
                padding: 0.75rem 1rem;
                border-radius: 14px;
                border: none;
                box-shadow: 0 0 18px rgba(124, 58, 237, 0.55);
                transition: transform 0.2s ease, box-shadow 0.3s ease;
                margin-top: 0.6rem;
            }

            div.stButton > button:hover {
                transform: translateY(-2px) scale(1.01);
                box-shadow: 0 0 28px rgba(124, 58, 237, 0.85), 0 0 40px rgba(37, 99, 235, 0.4);
                color: white;
            }

            div.stButton > button:active {
                transform: translateY(0) scale(0.99);
            }

            /* ---------- Result cards ---------- */
            .result-card {
                border-radius: 22px;
                padding: 2rem 1.5rem;
                text-align: center;
                margin-top: 1.5rem;
                animation: fadeIn 0.5s ease-out, pulseGlow 2.4s ease-in-out infinite;
            }

            .result-card.spam {
                background: rgba(239, 68, 68, 0.12);
                border: 1px solid rgba(239, 68, 68, 0.45);
                box-shadow: 0 0 30px rgba(239, 68, 68, 0.25);
            }

            .result-card.ham {
                background: rgba(52, 211, 153, 0.12);
                border: 1px solid rgba(52, 211, 153, 0.45);
                box-shadow: 0 0 30px rgba(52, 211, 153, 0.25);
            }

            @keyframes pulseGlow {
                0%, 100% { box-shadow: 0 0 22px rgba(255,255,255,0.08); }
                50% { box-shadow: 0 0 36px rgba(255,255,255,0.18); }
            }

            .result-icon {
                font-size: 3.2rem;
                margin-bottom: 0.4rem;
                display: block;
            }

            .result-label {
                font-family: 'Poppins', sans-serif;
                font-weight: 700;
                font-size: 1.8rem;
                margin-bottom: 0.2rem;
            }

            .result-card.spam .result-label { color: #f87171; }
            .result-card.ham .result-label { color: #6ee7b7; }

            .result-desc {
                color: #cfcbe8;
                font-size: 0.95rem;
            }

            /* ---------- Confidence bars ---------- */
            .conf-row {
                margin-top: 0.9rem;
            }

            .conf-label {
                display: flex;
                justify-content: space-between;
                color: #d9d6f5;
                font-size: 0.92rem;
                margin-bottom: 0.3rem;
                font-weight: 500;
            }

            .stProgress > div > div {
                border-radius: 8px !important;
            }

            /* ---------- Footer ---------- */
            .app-footer {
                text-align: center;
                margin-top: 2.5rem;
                padding-top: 1.2rem;
                border-top: 1px solid rgba(255,255,255,0.08);
                color: #8b85b3;
                font-size: 0.85rem;
            }

            .footer-badges {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.7rem;
            }

            .badge {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 999px;
                padding: 0.3rem 0.85rem;
                font-size: 0.78rem;
                color: #cfcbe8;
                transition: all 0.25s ease;
            }

            .badge:hover {
                background: rgba(167, 139, 250, 0.18);
                border-color: rgba(167, 139, 250, 0.5);
                transform: translateY(-2px);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# UI sections
# --------------------------------------------------------------------------
def render_header():
    """Render the application header and subtitle."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">🛡 Spam Email Classifier</div>
            <div class="app-subtitle">AI-powered SMS &amp; Email Spam Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_card():
    """Render the message input card and return the user's text and the
    predict button's clicked state.

    Returns:
        A tuple of (user_text, predict_clicked).
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3>✉️ Enter a message</h3>", unsafe_allow_html=True)

    user_text = st.text_area(
        label="Message input",
        placeholder="Paste your email or SMS...",
        height=180,
        label_visibility="collapsed",
    )

    predict_clicked = st.button("🔍 Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    return user_text, predict_clicked


def render_result_card(prediction):
    """Render the spam/ham verdict card.

    Args:
        prediction: The predicted label, either "spam" or "ham".
    """
    if prediction == "spam":
        st.markdown(
            """
            <div class="result-card spam">
                <span class="result-icon">⚠️</span>
                <div class="result-label">SPAM DETECTED</div>
                <div class="result-desc">This message shows strong signs of being spam.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card ham">
                <span class="result-icon">✅</span>
                <div class="result-label">LOOKS SAFE</div>
                <div class="result-desc">This message appears to be legitimate (ham).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_confidence_card(spam_prob, ham_prob):
    """Render the confidence breakdown card with animated progress bars.

    Args:
        spam_prob: Probability of the "spam" class (0-1) or None.
        ham_prob: Probability of the "ham" class (0-1) or None.
    """
    if spam_prob is None or ham_prob is None:
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Confidence Breakdown</h3>", unsafe_allow_html=True)

    dominant_pct = max(spam_prob, ham_prob) * 100

    st.markdown(
        f"""
        <div class="conf-label">
            <span>Overall Confidence</span>
            <span>{dominant_pct:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(dominant_pct / 100, 0.0), 1.0))

    st.markdown('<div class="conf-row">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="conf-label">
            <span>🚫 Spam probability</span>
            <span>{spam_prob * 100:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(spam_prob, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="conf-row">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="conf-label">
            <span>✅ Ham probability</span>
            <span>{ham_prob * 100:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(ham_prob, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    """Render the footer with technology badges."""
    st.markdown(
        """
        <div class="app-footer">
            Built with a scikit-learn TF-IDF + Naive Bayes pipeline
            <div class="footer-badges">
                <span class="badge">Python</span>
                <span class="badge">Scikit-Learn</span>
                <span class="badge">TF-IDF</span>
                <span class="badge">Naive Bayes</span>
                <span class="badge">Streamlit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
def main():
    """Entry point for the Streamlit application."""
    inject_custom_css()
    render_header()

    # Load model and vectorizer with graceful error handling.
    try:
        model = load_model()
        vectorizer = load_vectorizer()
    except FileNotFoundError as error:
        st.error(f"❌ Failed to load required files: {error}")
        st.stop()
    except Exception as error:  # noqa: BLE001 - surface any unexpected load error
        st.error(f"❌ An unexpected error occurred while loading resources: {error}")
        st.stop()

    user_text, predict_clicked = render_input_card()

    if predict_clicked:
        if not user_text or not user_text.strip():
            st.warning("⚠️ Please enter a message before predicting.")
        else:
            with st.spinner("Analyzing message..."):
                time.sleep(0.4)  # brief delay so the spinner is visible
                try:
                    prediction, spam_prob, ham_prob = predict_message(
                        user_text, model, vectorizer
                    )
                except Exception as error:  # noqa: BLE001 - surface prediction errors
                    st.error(f"❌ Prediction failed: {error}")
                    render_footer()
                    return

            render_result_card(prediction)
            render_confidence_card(spam_prob, ham_prob)

    render_footer()


if __name__ == "__main__":
    main()
