"""
risk_engine.py

Combines the spam classifier's output with the URL risk analysis from
url_checker.py into a single, unified assessment:
    - risk_score          (0-100)
    - risk_category       (Safe / Low Risk / Spam / Suspicious / High Risk /
                            Critical Risk — from config.RISK_BANDS)
    - final_prediction    (Safe / Spam / Phishing / Malicious / Critical Risk)
    - prediction_explanation (list of human-readable reasons)

This is the module that fixes the bug where prediction depended only on
the spam classifier's text score and ignored URL content entirely. A
message with entirely benign wording but a malicious/impersonating URL
must now be flagged, and vice versa.
"""

from src.config import RISK_BANDS
from src.url_checker import analyze_urls

# --------------------------------------------------------------------------
# Combination weights
# --------------------------------------------------------------------------
# Both channels (text-based spam signal, URL-based phishing signal) can
# independently justify a high risk_score — a malicious URL wrapped in
# innocuous-sounding text is exactly the "Amazon account locked" attack
# pattern this module exists to catch. So the base score is the STRONGER
# of the two signals, not a weighted average (an average would let a
# clean-sounding spam probability dilute a dangerous URL finding).
#
# A synergy bonus is added on top when BOTH channels show meaningful risk,
# since a suspicious link AND urgent/spammy language together is stronger
# evidence than either alone.
_SYNERGY_BONUS_FACTOR = 0.30

# Flag substrings that indicate a genuine phishing pattern (as opposed to
# generic spam) — used to decide the final_prediction label.
_PHISHING_INDICATOR_SUBSTRINGS = (
    "impersonation",
    "typosquatting",
    "raw ip address",
    "credential",
    "punycode",
    "url shortening",
    "url uses a raw ip address",
)


def _categorize_risk_score(risk_score: int) -> str:
    """Map a numeric risk_score to a named band using config.RISK_BANDS.

    Args:
        risk_score: Integer 0-100.

    Returns:
        The band name (e.g. "High Risk") whose range contains risk_score.
        Falls back to "Critical Risk" if score exceeds all defined bands
        (should not happen given risk_score is always clamped to 0-100).
    """
    for band_name, (low, high) in RISK_BANDS.items():
        if low <= risk_score <= high:
            return band_name
    return "Critical Risk"


def _looks_like_phishing(all_flags: list[str]) -> bool:
    """Decide whether the URL flags indicate phishing specifically.

    Args:
        all_flags: Flattened flag list from analyze_urls().

    Returns:
        True if any flag matches a known phishing-pattern indicator
        (brand impersonation, typosquatting, IP-literal URL, or
        credential-hiding tricks).
    """
    lowered_flags = [flag.lower() for flag in all_flags]
    return any(
        indicator.lower() in flag
        for flag in lowered_flags
        for indicator in _PHISHING_INDICATOR_SUBSTRINGS
    )


def _determine_final_prediction(
    spam_prediction: str,
    risk_score: int,
    url_all_flags: list[str],
) -> str:
    """Decide the final_prediction category label.

    Args:
        spam_prediction: Raw output of the spam classifier ("spam"/"ham").
        risk_score: Combined 0-100 risk score.
        url_all_flags: Flattened flags from the URL analysis.

    Returns:
        One of: "Safe", "Spam", "Phishing", "Malicious", "Critical Risk".
    """
    def _determine_final_prediction(
    spam_prediction: str,
    risk_score: int,
    url_all_flags: list[str],
) -> str:
     """Determine the final prediction."""

    if risk_score >= 90:
        return "Critical Risk"

    if _looks_like_phishing(url_all_flags) and risk_score >= 55:
        return "Phishing"

    if risk_score >= 75:
        return "Malicious"

    if spam_prediction == "spam" or risk_score >= 40:
        return "Spam"

    return "Safe"


def _build_explanation(
    spam_prediction: str,
    spam_probability: float,
    url_analysis: dict,
    risk_score: int,
    final_prediction: str,
) -> list[str]:
    """Build the human-readable prediction_explanation list.

    Combines the spam model's contribution with every URL-level flag so
    the user sees WHY a message was scored the way it was, not just a
    bare label — matching the "always explain WHY" requirement.

    Args:
        spam_prediction: "spam" or "ham".
        spam_probability: Probability (0-1) the spam model assigned to
            its predicted class.
        url_analysis: Output of analyze_urls().
        risk_score: Final combined risk score.
        final_prediction: Final category label.

    Returns:
        Ordered list of explanation strings.
    """
    explanation: list[str] = []

    if spam_prediction == "spam":
        explanation.append(
            f"Text content was classified as spam by the language model "
            f"(confidence {spam_probability * 100:.1f}%)."
        )
    else:
        explanation.append(
            f"Text content alone did not appear spam-like "
            f"(model confidence in 'ham': {spam_probability * 100:.1f}%)."
        )

    if url_analysis["url_count"] == 0:
        explanation.append("No URLs were found in the message.")
    else:
        explanation.append(
            f"Found {url_analysis['url_count']} URL(s); highest single-URL "
            f"risk score was {url_analysis['max_url_risk']}/100."
        )
        if url_analysis["max_phishing_ml_probability"] is not None:
            explanation.append(
                f"ML phishing_classifier probability (highest across URLs): "
                f"{url_analysis['max_phishing_ml_probability'] * 100:.1f}%."
            )
        explanation.extend(url_analysis["all_flags"])

    explanation.append(
        f"Combined risk score: {risk_score}/100 -> classified as '{final_prediction}'."
    )

    return explanation


def compute_risk_score(spam_probability: float, url_max_risk: int) -> tuple[int, int]:
    """Combine the spam and URL risk channels into a single 0-100 score.

    Args:
        spam_probability: Probability (0-1) the spam model assigned to its
            predicted class. Only meaningful as a risk signal when the
            predicted class is "spam" — callers should pass 0.0 here if
            the predicted class was "ham".
        url_max_risk: The highest single-URL risk_score (0-100) found by
            url_checker.analyze_urls() (already blends heuristic + ML
            phishing_classifier signals internally).

    Returns:
        Tuple of (risk_score, confidence), both integers clamped to
        [0, 100]. confidence is the strength of the single strongest
        contributing signal, before the synergy bonus is added — i.e. how
        sure the engine is based on its best individual piece of evidence,
        independent of whether a second, weaker signal also agreed.
    """
    spam_component = round(spam_probability * 100)
    url_component = url_max_risk

    base_score = max(spam_component, url_component)
    weaker_signal = min(spam_component, url_component)
    synergy_bonus = round(weaker_signal * _SYNERGY_BONUS_FACTOR)

    risk_score = min(100, base_score + synergy_bonus)
    confidence = base_score

    return risk_score, confidence


def analyze_email_risk(
    spam_prediction: str,
    spam_probability: float,
    url_list: list[str],
) -> dict:
    """Run the full risk engine: URL analysis + score combination + explanation.

    This is the single entry point predict.py (and the Streamlit app)
    should call once the spam classifier has produced its prediction.

    Args:
        spam_prediction: "spam" or "ham", from the spam classifier.
        spam_probability: Probability (0-1) of the PREDICTED class (i.e.
            model.predict_proba(...).max()). If spam_prediction is "ham",
            this value is not used as risk signal (treated as 0 risk
            contribution from the text channel).
        url_list: URLs extracted from the raw email body via
            url_features.extract_urls().

    Returns:
        prediction_result dict with keys:
            final_prediction       - category label (str)
            risk_score              - combined 0-100 score (int)
            risk_category           - named band from config.RISK_BANDS
            spam_prediction         - passthrough of input
            spam_probability        - passthrough of input
            url_analysis            - full output of analyze_urls()
            prediction_explanation  - list[str] explaining the verdict
    """
    url_analysis = analyze_urls(url_list)

    # Only count the spam probability as a risk contributor when the
    # model actually predicted "spam" — a 95%-confident "ham" prediction
    # should not itself add risk.
    effective_spam_probability = spam_probability if spam_prediction == "spam" else 0.0

    risk_score, confidence = compute_risk_score(effective_spam_probability, url_analysis["max_url_risk"])
    risk_category = _categorize_risk_score(risk_score)
    final_prediction = _determine_final_prediction(
        spam_prediction, risk_score, url_analysis["all_flags"]
    )
    explanation = _build_explanation(
        spam_prediction, spam_probability, url_analysis, risk_score, final_prediction
    )

    return {
        "final_prediction": final_prediction,
        "risk_score": risk_score,
        "confidence": confidence,
        "risk_category": risk_category,
        "spam_prediction": spam_prediction,
        "spam_probability": spam_probability,
        "url_analysis": url_analysis,
        "prediction_explanation": explanation,
    }
