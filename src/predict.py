"""
predict.py

The full prediction pipeline: text preprocessing + spam classification +
URL extraction + URL risk analysis + combined risk scoring.

This is the single integrated entry point other code (the Streamlit app,
report generation, etc.) should call to get a complete assessment of an
email. It replaces the earlier version of this file, which only ran the
spam classifier and never looked at URLs at all — the root cause of the
"Amazon account locked" phishing email being scored as Safe.
"""

import joblib

from src.config import SPAM_CLASSIFIER_PATH, SPAM_POSITIVE_LABEL, TFIDF_VECTORIZER_PATH
from src.preprocess import preprocess_text
from src.risk_engine import analyze_email_risk
from src.url_features import extract_urls
from src.utils import ensure_file_exists, get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Model / vectorizer loading
# --------------------------------------------------------------------------
ensure_file_exists(SPAM_CLASSIFIER_PATH, "Spam classifier model")
ensure_file_exists(TFIDF_VECTORIZER_PATH, "TF-IDF vectorizer")

model = joblib.load(SPAM_CLASSIFIER_PATH)
tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

logger.info("Spam classifier and vectorizer loaded successfully.")


def classify_spam(email_body: str) -> tuple[str, float]:
    """Run the spam classifier on raw email text.

    Args:
        email_body: Raw, unprocessed email text.

    Returns:
        A tuple of (prediction_result, prediction_probability):
            prediction_result      - "spam" or "ham"
            prediction_probability - probability (0-1) of the PREDICTED
                                      class (i.e. the model's confidence
                                      in whichever label it chose)
    """
    processed_text = preprocess_text(email_body)
    feature_matrix = tfidf_vectorizer.transform([processed_text])

    prediction_result = model.predict(feature_matrix)[0]
    prediction_probability = float(model.predict_proba(feature_matrix).max())

    return prediction_result, prediction_probability


def analyze_email(email_body: str) -> dict:
    """Run the complete, integrated email security pipeline.

    Pipeline order (matches the architecture in the project README):
        1. Spam classification on preprocessed text.
        2. URL extraction from the RAW (unprocessed) email body — this
           must happen before preprocessing strips punctuation, or
           "https://..." becomes unrecognizable.
        3. URL feature extraction + reputation checking.
        4. Combination of both signals into a single risk_score.
        5. Human-readable explanation of the verdict.

    Args:
        email_body: Raw email or message text, exactly as received.

    Returns:
        prediction_result dict (see risk_engine.analyze_email_risk for the
        full schema), plus the two top-level fields:
            email_body  - the original input, for reference in the UI/logs
            url_list    - URLs extracted from email_body
    """
    if not email_body or not email_body.strip():
        raise ValueError("email_body must be non-empty.")

    spam_prediction, spam_probability = classify_spam(email_body)
    url_list = extract_urls(email_body)

    result = analyze_email_risk(
        spam_prediction=spam_prediction,
        spam_probability=spam_probability,
        url_list=url_list,
    )

    result["email_body"] = email_body
    result["url_list"] = url_list

    logger.info(
        "Analyzed email: spam=%s (%.1f%%), urls=%d, risk_score=%d, final=%s",
        spam_prediction,
        spam_probability * 100,
        len(url_list),
        result["risk_score"],
        result["final_prediction"],
    )

    return result


def predict_message(message: str) -> None:
    """CLI-facing prediction: run the full pipeline and print a report.

    Kept as the original entry point for command-line use so existing
    workflows (`python -m src.predict`) continue to work, now backed by
    the full integrated pipeline instead of spam-only classification.

    Args:
        message: Raw message/email text to analyze.
    """
    result = analyze_email(message)

    print("\n========== RESULT ==========\n")
    print(f"Message           : {message}")
    print(f"Spam Prediction   : {result['spam_prediction'].upper()} "
          f"({result['spam_probability'] * 100:.2f}% confidence)")
    print(f"URLs Found        : {result['url_list'] or 'None'}")
    print(f"Risk Score        : {result['risk_score']}/100")
    print(f"Risk Category     : {result['risk_category']}")
    print(f"Final Prediction  : {result['final_prediction']}")
    print("\nExplanation:")
    for line in result["prediction_explanation"]:
        print(f"  - {line}")


if __name__ == "__main__":

    print("========== AI Email Security Assistant ==========\n")

    while True:

        user_message = input("Enter your message (paste full email, URLs included): ")

        predict_message(user_message)

        choice = input("\nDo you want to test another message? (y/n): ").lower()

        if choice != "y":
            print("\nThank you for using the Email Security Assistant!")
            break
