"""
url_checker.py

Applies reputation/judgment heuristics to the objective features produced
by url_features.py. Detects:
    - Suspicious top-level domains commonly abused in phishing campaigns
    - URL shorteners (used to hide the true destination)
    - Brand impersonation / typosquatting against a list of known brands
    - Suspicious keywords in the URL (login, verify, secure, etc.)
    - IP-literal URLs, missing HTTPS, unusually long URLs, credential-
      hiding "@" tricks, and high-entropy (likely auto-generated) domains

Each check contributes points toward a 0-100 risk_score for that single
URL, plus a human-readable flag explaining why. url_features.py measures;
this module decides what those measurements mean.
"""

import difflib

import joblib
import pandas as pd

from src.config import PHISHING_CLASSIFIER_PATH
from src.url_features import KNOWN_SHORTENER_DOMAINS, extract_full_feature_vector, extract_url_features
from src.utils import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# ML phishing model (trained by phishing_model.py on the Vrbančič et al.
# Phishing-Dataset — see README.md for full citation)
# --------------------------------------------------------------------------
# Loaded once at import time, like src/predict.py does for the spam model.
# Loading is wrapped in a try/except rather than using ensure_file_exists,
# because url_checker.py must still function with heuristics alone if the
# phishing model hasn't been trained yet (e.g. a fresh clone of the repo
# before running `python -m src.phishing_model`) — graceful degradation
# instead of a hard crash.
_PHISHING_FEATURE_COLUMNS_PATH = PHISHING_CLASSIFIER_PATH.parent / "phishing_feature_columns.joblib"

try:
    _phishing_pipeline = joblib.load(PHISHING_CLASSIFIER_PATH)
    _phishing_feature_columns = joblib.load(_PHISHING_FEATURE_COLUMNS_PATH)
    logger.info("phishing_classifier loaded successfully.")
except FileNotFoundError:
    _phishing_pipeline = None
    _phishing_feature_columns = None
    logger.warning(
        "phishing_classifier not found — URL analysis will use heuristics "
        "only. Run `python -m src.phishing_model` to train it."
    )


def get_phishing_ml_probability(url: str) -> float | None:
    """Score a URL with the trained ML phishing_classifier.

    Args:
        url: A URL string.

    Returns:
        Probability (0-1) that the model assigns to the "phishing" class,
        or None if phishing_classifier has not been trained/loaded.
    """
    if _phishing_pipeline is None:
        return None

    feature_vector = extract_full_feature_vector(url)
    # Build a single-row DataFrame with columns in the EXACT order the
    # model was trained on — column order mismatches silently produce
    # garbage predictions with scikit-learn, so this alignment is critical.
    row = pd.DataFrame([feature_vector])[_phishing_feature_columns]

    probability = _phishing_pipeline.predict_proba(row)[0][1]
    return float(probability)

# --------------------------------------------------------------------------
# Reputation reference data
# --------------------------------------------------------------------------
# TLDs disproportionately associated with spam/phishing campaigns, largely
# because they are cheap or free to register with minimal verification.
# This is a heuristic list, not exhaustive — legitimate sites do exist on
# some of these, which is why this contributes points rather than an
# automatic verdict.
SUSPICIOUS_TLDS: frozenset[str] = frozenset({
    "xyz", "top", "club", "tk", "ml", "ga", "cf", "gq", "work", "click",
    "link", "loan", "win", "review", "download", "stream", "men", "party",
    "science", "racing", "accountant", "faith", "date", "bid", "trade",
    "webcam", "cricket", "kim", "country", "gdn",
})

# Words that frequently appear in phishing URLs, particularly in the path
# or query string, designed to create urgency or imitate legitimate
# account-management flows.
SUSPICIOUS_KEYWORDS: frozenset[str] = frozenset({
    "login", "verify", "secure", "account", "update", "confirm", "signin",
    "banking", "suspended", "locked", "password", "security", "billing",
    "invoice", "urgent", "restore", "unlock", "validate", "authenticate",
    "webscr", "recovery",
})

# Well-known brands frequently impersonated in phishing campaigns, mapped
# to their legitimate registrable domain. Used to detect both direct
# impersonation (brand name embedded in a non-official domain) and
# typosquatting (a domain label that's suspiciously *similar* to the brand
# without being an exact match).
KNOWN_BRANDS: dict[str, str] = {
    "amazon": "amazon.com",
    "paypal": "paypal.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "netflix": "netflix.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "linkedin": "linkedin.com",
    "ebay": "ebay.com",
    "chase": "chase.com",
    "wellsfargo": "wellsfargo.com",
    "bankofamerica": "bankofamerica.com",
    "dropbox": "dropbox.com",
    "docusign": "docusign.com",
}

# --------------------------------------------------------------------------
# Scoring weights (points added toward a single URL's 0-100 risk_score)
# --------------------------------------------------------------------------
_WEIGHTS: dict[str, int] = {
    "no_https": 10,
    "ip_address": 30,
    "suspicious_tld": 20,
    "url_shortener": 15,
    "brand_impersonation": 60,
    "typosquatting": 60,
    "suspicious_keyword": 8,   # per keyword found, capped below
    "at_symbol": 20,
    "long_url": 10,
    "excessive_subdomains": 10,
    "high_entropy": 15,
    "punycode": 25,
}
_MAX_KEYWORD_POINTS = 24  # cap so keyword-stuffing can't dominate the score
_LONG_URL_THRESHOLD = 75
_EXCESSIVE_SUBDOMAIN_THRESHOLD = 3
_HIGH_ENTROPY_THRESHOLD = 3.8


def _check_brand_impersonation(domain: str) -> str | None:
    """Detect a known brand name embedded in a non-official domain.

    Example: "amazon-security-login.xyz" contains "amazon" but is not
    amazon.com or a subdomain of it.

    Args:
        domain: Lowercased domain string.

    Returns:
        A flag message if impersonation is detected, otherwise None.
    """
    for brand, official_domain in KNOWN_BRANDS.items():
        if brand not in domain:
            continue
        if domain == official_domain or domain.endswith(f".{official_domain}"):
            continue
        return (
            f"Domain contains brand name '{brand}' but is not an official "
            f"{official_domain} domain (possible brand impersonation)"
        )
    return None


def _check_typosquatting(domain: str) -> str | None:
    """Detect a domain label that's suspiciously similar to a known brand.

    Uses character-level similarity (difflib) rather than exact substring
    matching, catching lookalikes like "arnazon.com" or "paypa1.com" that
    _check_brand_impersonation would miss.

    Args:
        domain: Lowercased domain string.

    Returns:
        A flag message if likely typosquatting is detected, otherwise None.
    """
    labels = [label for label in domain.split(".") if label]
    if not labels:
        return None

    # The registrable label is normally the second-to-last part
    # (example.com -> "example"); for a bare host just use the first label.
    primary_label = labels[-2] if len(labels) >= 2 else labels[0]

    for brand in KNOWN_BRANDS:
        if primary_label == brand:
            continue  # exact match is legitimate, not typosquatting
        similarity = difflib.SequenceMatcher(None, primary_label, brand).ratio()
        if similarity >= 0.75:
            return (
                f"Domain label '{primary_label}' closely resembles known "
                f"brand '{brand}' (similarity {similarity:.0%}) — possible "
                f"typosquatting"
            )
    return None


def _check_punycode(domain: str) -> str | None:
    """Detect punycode-encoded (IDN) domain labels.

    Punycode ("xn--" prefix) is the standard ASCII encoding for
    internationalized domain names. It's legitimate in general, but is
    also the mechanism behind homograph attacks — e.g. registering a
    domain with a Cyrillic 'а' that displays identically to Latin 'a' in
    "amazon.com", which browsers show as "xn--mazon-...". A domain using
    punycode alongside other risk signals is worth flagging explicitly.

    Args:
        domain: Lowercased domain string.

    Returns:
        A flag message if any label starts with "xn--", otherwise None.
    """
    for label in domain.split("."):
        if label.startswith("xn--"):
            return (
                "Domain uses punycode encoding (xn--), which can hide "
                "look-alike international characters mimicking a trusted brand"
            )
    return None


def check_url(url: str) -> dict:
    """Run all reputation heuristics against a single URL.

    Args:
        url: A URL string.

    Returns:
        A dict with:
            url          - the original URL
            risk_score   - 0-100 heuristic risk score for this URL alone
            flags        - list of human-readable reasons contributing to
                            the score (used directly in prediction_explanation)
            features     - the underlying feature dict from url_features.py
    """
    features = extract_url_features(url)
    flags: list[str] = []
    score = 0

    if not features["uses_https"]:
        flags.append("URL does not use HTTPS")
        score += _WEIGHTS["no_https"]

    if features["has_ip_address"]:
        flags.append("URL uses a raw IP address instead of a domain name")
        score += _WEIGHTS["ip_address"]

    if features["tld"] in SUSPICIOUS_TLDS:
        flags.append(f"Uses a top-level domain commonly abused in phishing (.{features['tld']})")
        score += _WEIGHTS["suspicious_tld"]

    if features["domain"] in KNOWN_SHORTENER_DOMAINS:
        flags.append(f"Uses a URL shortening service ({features['domain']}) that hides the real destination")
        score += _WEIGHTS["url_shortener"]

    impersonation_flag = _check_brand_impersonation(features["domain"])
    if impersonation_flag:
        flags.append(impersonation_flag)
        score += _WEIGHTS["brand_impersonation"]
    else:
        typosquat_flag = _check_typosquatting(features["domain"])
        if typosquat_flag:
            flags.append(typosquat_flag)
            score += _WEIGHTS["typosquatting"]

    punycode_flag = _check_punycode(features["domain"])
    if punycode_flag:
        flags.append(punycode_flag)
        score += _WEIGHTS["punycode"]

    if features["has_at_symbol"]:
        flags.append("URL contains '@', a technique used to disguise the real destination")
        score += _WEIGHTS["at_symbol"]

    if features["url_length"] > _LONG_URL_THRESHOLD:
        flags.append(f"URL is excessively long ({features['url_length']} characters)")
        score += _WEIGHTS["long_url"]

    if features["num_subdomains"] > _EXCESSIVE_SUBDOMAIN_THRESHOLD:
        flags.append(f"URL has an unusually high number of subdomains ({features['num_subdomains']})")
        score += _WEIGHTS["excessive_subdomains"]

    if features["domain_entropy"] > _HIGH_ENTROPY_THRESHOLD:
        flags.append("Domain name has high randomness, suggesting auto-generated infrastructure")
        score += _WEIGHTS["high_entropy"]

    full_url_lower = url.lower()
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in full_url_lower]
    if found_keywords:
        keyword_points = min(len(found_keywords) * _WEIGHTS["suspicious_keyword"], _MAX_KEYWORD_POINTS)
        flags.append(f"Contains suspicious keyword(s): {', '.join(sorted(found_keywords))}")
        score += keyword_points

    heuristic_score = min(score, 100)

    # --- ML model signal (phishing_classifier, trained on real data) ---
    phishing_ml_probability = get_phishing_ml_probability(url)
    ml_score = round(phishing_ml_probability * 100) if phishing_ml_probability is not None else 0

    if phishing_ml_probability is not None and phishing_ml_probability >= 0.5:
        flags.append(
            f"ML phishing model predicts {phishing_ml_probability * 100:.1f}% "
            f"probability this URL is phishing (trained on real-world phishing/legitimate URLs)"
        )

    # KNOWN LIMITATION: the training dataset's directory/file/parameter
    # feature-extraction algorithm was never published (only feature
    # DEFINITIONS were, in the dataset's README) — url_features.py
    # reconstructs it from those definitions as best it can, but testing
    # showed this approximation makes phishing_classifier noisy on
    # ordinary URLs with a path or query string but NO other risk
    # indicator (e.g. a plain "company.com/reports?id=42" scored ~77%
    # "phishing" from the ML signal alone). Rather than ship that false-
    # positive risk, the ML score's standalone influence is capped
    # whenever heuristic_score found zero corroborating evidence — it can
    # still push a URL into "worth a second look" territory, but not
    # single-handedly brand a clean-looking URL "Malicious". When ANY
    # heuristic flag also fires, the ML score is trusted at full strength,
    # since the two signals corroborating each other is exactly the kind
    # of agreement a security tool should escalate on.
    _UNCORROBORATED_ML_CAP = 30

    if heuristic_score == 0:
        ml_contribution = min(ml_score, _UNCORROBORATED_ML_CAP)
    else:
        ml_contribution = ml_score

    # Combine heuristic and ML scores the same way risk_engine.py combines
    # spam-text and URL-channel signals: take the stronger signal as the
    # base, then add a smaller synergy bonus when both agree, so a single
    # confident channel is never diluted by averaging with a weaker one.
    base = max(heuristic_score, ml_contribution)
    synergy_bonus = round(min(heuristic_score, ml_contribution) * 0.3)
    combined_score = min(100, base + synergy_bonus)

    return {
        "url": url,
        "risk_score": combined_score,
        "heuristic_score": heuristic_score,
        "phishing_ml_probability": phishing_ml_probability,
        "flags": flags,
        "features": features,
    }


def analyze_urls(url_list: list[str]) -> dict:
    """Run check_url() over every URL and aggregate the results.

    Args:
        url_list: URLs extracted from an email body.

    Returns:
        A dict with:
            url_count           - number of URLs analyzed
            per_url_results     - list of check_url() results, one per URL
            max_url_risk        - highest single-URL combined risk_score
                                   found (0 if no URLs present)
            max_phishing_ml_probability - highest ML phishing_classifier
                                   probability found across URLs, or None
                                   if the model isn't loaded
            all_flags           - flattened, de-duplicated list of every
                                   flag raised across all URLs
    """
    if not url_list:
        return {
            "url_count": 0,
            "per_url_results": [],
            "max_url_risk": 0,
            "max_phishing_ml_probability": None,
            "all_flags": [],
        }

    per_url_results = [check_url(url) for url in url_list]
    max_url_risk = max(result["risk_score"] for result in per_url_results)

    ml_probabilities = [
        result["phishing_ml_probability"]
        for result in per_url_results
        if result["phishing_ml_probability"] is not None
    ]
    max_phishing_ml_probability = max(ml_probabilities) if ml_probabilities else None

    all_flags: list[str] = []
    for result in per_url_results:
        for flag in result["flags"]:
            if flag not in all_flags:
                all_flags.append(flag)

    return {
        "url_count": len(url_list),
        "per_url_results": per_url_results,
        "max_url_risk": max_url_risk,
        "max_phishing_ml_probability": max_phishing_ml_probability,
        "all_flags": all_flags,
    }
