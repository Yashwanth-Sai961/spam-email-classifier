"""
Streamlit UI for the AI Email Security Assistant.

This module ONLY handles presentation (UI/UX). The underlying pipeline —
spam classification, URL extraction, URL risk analysis, and combined risk
scoring — lives entirely in src/predict.py (which itself composes
src/preprocess.py, src/url_features.py, src/url_checker.py, and
src/risk_engine.py). This file never re-implements or duplicates that
logic; it calls src.predict.analyze_email() and renders the result.

Project structure assumptions (unchanged):
    models/spam_classifier.joblib
    models/tfidf_vectorizer.joblib
    src/predict.py -> analyze_email()
"""

import os
import sys
import time

import streamlit as st

# --------------------------------------------------------------------------
# Path setup
# --------------------------------------------------------------------------
# BASE_DIR points to the project root (one level above this file's folder,
# i.e. .../spam_email_classifier)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure the project root is importable so `from src.predict import
# analyze_email` works regardless of which directory streamlit was
# launched from.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import analyze_email, analyze_email_file  # noqa: E402  (import after sys.path setup)


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

            .result-card.spam,
            .result-card.malicious {
                background: rgba(239, 68, 68, 0.12);
                border: 1px solid rgba(239, 68, 68, 0.45);
                box-shadow: 0 0 30px rgba(239, 68, 68, 0.25);
            }

            .result-card.ham,
            .result-card.safe {
                background: rgba(52, 211, 153, 0.12);
                border: 1px solid rgba(52, 211, 153, 0.45);
                box-shadow: 0 0 30px rgba(52, 211, 153, 0.25);
            }

            .result-card.phishing,
            .result-card.critical-risk {
                background: rgba(239, 68, 68, 0.18);
                border: 1px solid rgba(248, 113, 113, 0.65);
                box-shadow: 0 0 36px rgba(239, 68, 68, 0.4);
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

            .result-card.spam .result-label,
            .result-card.malicious .result-label { color: #f87171; }
            .result-card.ham .result-label,
            .result-card.safe .result-label { color: #6ee7b7; }
            .result-card.phishing .result-label,
            .result-card.critical-risk .result-label { color: #fca5a5; }

            .result-desc {
                color: #cfcbe8;
                font-size: 0.95rem;
            }

            /* ---------- URL analysis ---------- */
            .url-entry {
                background: rgba(0, 0, 0, 0.22);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 14px;
                padding: 1rem 1.2rem;
                margin-top: 0.8rem;
            }

            .url-entry-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 0.5rem;
            }

            .url-text {
                font-family: monospace;
                color: #cbd5ff;
                word-break: break-all;
                font-size: 0.9rem;
            }

            .url-risk-badge {
                font-family: 'Poppins', sans-serif;
                font-weight: 700;
                font-size: 0.85rem;
                padding: 0.25rem 0.7rem;
                border-radius: 999px;
                white-space: nowrap;
            }

            .url-risk-badge.low {
                background: rgba(52, 211, 153, 0.18);
                color: #6ee7b7;
                border: 1px solid rgba(52, 211, 153, 0.4);
            }

            .url-risk-badge.medium {
                background: rgba(251, 191, 36, 0.18);
                color: #fbbf24;
                border: 1px solid rgba(251, 191, 36, 0.4);
            }

            .url-risk-badge.high {
                background: rgba(239, 68, 68, 0.2);
                color: #f87171;
                border: 1px solid rgba(239, 68, 68, 0.5);
            }

            .url-flag-list {
                margin: 0.4rem 0 0 0;
                padding-left: 1.2rem;
                color: #d9d6f5;
                font-size: 0.88rem;
            }

            .url-flag-list li {
                margin-bottom: 0.25rem;
            }

            .explanation-list {
                margin: 0.5rem 0 0 0;
                padding-left: 1.2rem;
                color: #d9d6f5;
                font-size: 0.92rem;
            }

            .explanation-list li {
                margin-bottom: 0.4rem;
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
    uploaded_file = st.file_uploader(
    "Or upload an email (.eml)",
    type=["eml"],
    )
    predict_clicked = st.button("🔍 Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    return user_text, uploaded_file, predict_clicked


# Maps final_prediction values (from src.risk_engine) to the CSS class,
# icon, headline, and description shown on the verdict card. Centralized
# here so adding a new category later is a one-line change.
_PREDICTION_DISPLAY = {
    "Safe": ("safe", "✅", "LOOKS SAFE", "No significant spam or phishing indicators were found."),
    "Spam": ("spam", "🚫", "SPAM DETECTED", "This message shows strong signs of being spam."),
    "Phishing": ("phishing", "🎣", "PHISHING DETECTED", "This message contains a link designed to impersonate a trusted brand or steal credentials."),
    "Malicious": ("malicious", "☠️", "MALICIOUS CONTENT", "This message combines high-risk text and URL indicators."),
    "Critical Risk": ("critical-risk", "🔥", "CRITICAL RISK", "This message shows severe, multiple indicators of attack. Do not click any links."),
}


def render_result_card(result):
    """Render the overall verdict card for the combined risk assessment.

    Args:
        result: The prediction_result dict returned by
            src.predict.analyze_email() (final_prediction, risk_score,
            risk_category, etc.).
    """
    final_prediction = result["final_prediction"]
    css_class, icon, headline, description = _PREDICTION_DISPLAY.get(
        final_prediction, ("spam", "⚠️", final_prediction.upper(), "")
    )

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <span class="result-icon">{icon}</span>
            <div class="result-label">{headline}</div>
            <div class="result-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_score_card(result):
    """Render the combined risk score gauge and full explanation list.

    Args:
        result: The prediction_result dict from analyze_email(), containing
            risk_score, risk_category, spam_prediction, spam_probability,
            and prediction_explanation.
    """
    risk_score = result["risk_score"]
    risk_category = result["risk_category"]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Combined Risk Score</h3>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="conf-label">
            <span>Risk Score ({risk_category})</span>
            <span>{risk_score}/100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(risk_score / 100, 0.0), 1.0))

    st.markdown(
        f"""
        <div class="conf-label" style="margin-top:0.6rem;">
            <span>Confidence (strongest single signal)</span>
            <span>{result['confidence']}/100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="conf-row">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="conf-label">
            <span>🚫 Spam model confidence ({result['spam_prediction']})</span>
            <span>{result['spam_probability'] * 100:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(result["spam_probability"], 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top:1.2rem;'>🧠 Why this verdict</h3>", unsafe_allow_html=True)
    explanation_items = "".join(f"<li>{line}</li>" for line in result["prediction_explanation"])
    st.markdown(f'<ul class="explanation-list">{explanation_items}</ul>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _url_risk_bucket(url_risk_score):
    """Map a single URL's 0-100 risk_score to a low/medium/high CSS bucket."""
    if url_risk_score >= 60:
        return "high"
    if url_risk_score >= 30:
        return "medium"
    return "low"


def render_url_analysis_card(result):
    """Render the per-URL breakdown: domain, risk score, and flags.

    Only rendered when at least one URL was found in the message — a
    message with no URLs simply won't show this card, since there is
    nothing to analyze.

    Args:
        result: The prediction_result dict from analyze_email(), containing
            url_list and url_analysis (from src.url_checker.analyze_urls).
    """
    url_analysis = result["url_analysis"]

    if url_analysis["url_count"] == 0:
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f"<h3>🔗 URL Analysis ({url_analysis['url_count']} found)</h3>",
        unsafe_allow_html=True,
    )

    for url_result in url_analysis["per_url_results"]:
        bucket = _url_risk_bucket(url_result["risk_score"])
        flags_html = "".join(f"<li>{flag}</li>" for flag in url_result["flags"])
        flags_block = (
            f'<ul class="url-flag-list">{flags_html}</ul>'
            if flags_html
            else '<div class="result-desc" style="text-align:left;">No suspicious indicators found for this URL.</div>'
        )

        st.markdown(
            f"""
            <div class="url-entry">
                <div class="url-entry-header">
                    <span class="url-text">{url_result['url']}</span>
                    <span class="url-risk-badge {bucket}">{url_result['risk_score']}/100</span>
                </div>
                {flags_block}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    """Render the footer with technology badges."""
    st.markdown(
        """
        <div class="app-footer">
            Built with a scikit-learn TF-IDF + Naive Bayes text model, plus a heuristic URL &amp; phishing risk engine
            <div class="footer-badges">
                <span class="badge">Python</span>
                <span class="badge">Scikit-Learn</span>
                <span class="badge">TF-IDF</span>
                <span class="badge">Naive Bayes</span>
                <span class="badge">URL Risk Engine</span>
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

    user_text, uploaded_file, predict_clicked = render_input_card()
    if predict_clicked:

      if uploaded_file is None and (not user_text or not user_text.strip()):
        st.warning("⚠️ Please enter a message or upload a .eml file.")

      else:

        with st.spinner("Analyzing email..."):
            time.sleep(0.4)

            try:

                if uploaded_file is not None:
                    import tempfile

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as temp:
                        temp.write(uploaded_file.read())
                        temp_path = temp.name

                    result = analyze_email_file(temp_path)

                else:
                    result = analyze_email(user_text)

            except FileNotFoundError as error:
                st.error(f"❌ Failed to load required files: {error}")
                render_footer()
                return

            except Exception as error:
                st.error(f"❌ Analysis failed: {error}")
                render_footer()
                return

        render_result_card(result)
        render_risk_score_card(result)
        render_url_analysis_card(result)


    render_footer()


if __name__ == "__main__":
    main()
