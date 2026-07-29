# --------------------------------------------------------------------------
# Leetspeak / lookalike-character normalization
# --------------------------------------------------------------------------
# Phishing domains routinely swap letters for visually-similar digits/symbols
# (micr0soft, paypa1, g00gle, arnaz0n, etc.) to dodge naive substring checks
# while still looking convincing to a human at a glance. This map is used
# ONLY to build a comparison string — the original domain/URL is never
# altered, so flags/messages/scoring still reference the real domain.
import difflib
# --------------------------------------------------------------------------
# Known brands for phishing / impersonation detection
# --------------------------------------------------------------------------

KNOWN_BRANDS = {
    "paypal": "paypal.com",
    "microsoft": "microsoft.com",
    "google": "google.com",
    "amazon": "amazon.com",
    "apple": "apple.com",
    "netflix": "netflix.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "linkedin": "linkedin.com",
    "dropbox": "dropbox.com",
    "bankofamerica": "bankofamerica.com",
    "icici": "icicibank.com",
    "hdfc": "hdfcbank.com",
    "sbi": "sbi.co.in"
}


# ADD THIS
URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "buff.ly",
    "is.gd",
    "cutt.ly",
    "shorturl.at"
}
_LEET_SUBSTITUTIONS: dict[str, str] = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "9": "g", "@": "a", "$": "s",
}
_LEET_TRANSLATION_TABLE = str.maketrans(_LEET_SUBSTITUTIONS)


def _normalize_for_comparison(text: str) -> str:
    """Map lookalike digits/symbols to their letter equivalents for matching.

    Comparison-only helper — never used to rewrite the actual URL/domain
    that gets reported back to the caller.

    Args:
        text: Raw label text (already lowercased upstream).

    Returns:
        The text with leetspeak substitutions applied.
    """
    return text.translate(_LEET_TRANSLATION_TABLE)


def _get_comparable_labels(domain: str) -> list[str]:
    """Break a domain into every normalized label worth comparing to a brand.

    Domains are first split on "." (subdomain/registrable/TLD boundaries),
    then each of those parts is further split on "-", because modern
    phishing domains frequently smuggle a brand name into one hyphen-joined
    segment alongside a keyword, e.g. "micr0soft-security.com" ->
    ["micr0soft", "security", "com"]. Checking only the whole label or only
    the primary (registrable) label — as the previous implementation did —
    misses exactly this pattern. Each resulting piece is then leetspeak-
    normalized so "micr0soft" compares equal to "microsoft".

    Args:
        domain: Lowercased domain string.

    Returns:
        List of normalized label fragments (may contain duplicates; callers
        don't need uniqueness for correctness).
    """
    labels: list[str] = []
    for dot_label in domain.split("."):
        for hyphen_label in dot_label.split("-"):
            if hyphen_label:
                labels.append(_normalize_for_comparison(hyphen_label))
    return labels


# --------------------------------------------------------------------------
# Similarity metric for typosquatting: prefer real Levenshtein distance,
# fall back to difflib (stdlib-only) if the optional dependency isn't
# installed, so this module doesn't gain a hard new requirement.
# --------------------------------------------------------------------------
try:
    import Levenshtein

    def _similarity(a: str, b: str) -> float:
        """Return a 0-1 similarity ratio between two strings (Levenshtein)."""
        return Levenshtein.ratio(a, b)

except ImportError:
    def _similarity(a: str, b: str) -> float:
        """Return a 0-1 similarity ratio between two strings (difflib fallback)."""
        return difflib.SequenceMatcher(None, a, b).ratio()


def _is_legitimate_for_brand(domain: str, official_domain: str) -> bool:
    """True if `domain` IS, or is a genuine subdomain of, a brand's official domain.

    e.g. "microsoft.com" and "login.microsoft.com" are legitimate for the
    "microsoft" brand and must never be flagged, no matter what their
    individual labels look like after normalization.
    """
    return domain == official_domain or domain.endswith(f".{official_domain}")


def _check_brand_impersonation(domain: str) -> str | None:
    """Detect a known brand name embedded in a non-official domain.

    Unlike a single substring test against the whole domain, this now:
      1. Splits the domain into hyphen- and dot-separated labels, so a
         brand hidden in one segment of a multi-word domain (e.g. the
         "micr0soft" in "micr0soft-security.com") is still caught.
      2. Normalizes each label for common leetspeak substitutions first,
         so "micr0soft", "paypa1", "g00gle", "amaz0n" etc. are recognized
         as their real-word equivalents before comparison.

    Legitimate domains are still fully exempted: if `domain` is a brand's
    official domain or a subdomain of it, that brand is skipped entirely
    (checked against the raw domain, not the normalized labels, so a
    legitimate subdomain can never accidentally trip this check).

    Args:
        domain: Lowercased domain string.

    Returns:
        A flag message if impersonation is detected, otherwise None.
    """
    labels = _get_comparable_labels(domain)

    for brand, official_domain in KNOWN_BRANDS.items():
        if _is_legitimate_for_brand(domain, official_domain):
            continue  # genuinely the brand's own domain/subdomain — not impersonation

        for label in labels:
            if brand in label:
                return (
                    f"Domain contains brand name '{brand}' (found in label "
                    f"'{label}', possibly disguised with character "
                    f"substitutions) but is not an official {official_domain} "
                    f"domain (possible brand impersonation)"
                )
    return None


def _check_typosquatting(domain: str) -> str | None:
    """Detect a domain label that's suspiciously similar to a known brand.

    Every normalized label (post hyphen/dot-split, post leetspeak
    normalization) is compared against every known brand using edit-
    distance similarity (Levenshtein when available, difflib otherwise),
    rather than just the single primary label as before. This catches
    lookalikes such as "arnazon.com" or "paypa1-secure.com" regardless of
    which segment of the domain they appear in.

    A label that normalizes to an exact match for a brand is skipped here
    (that's handled — and already caught — by `_check_brand_impersonation`
    as impersonation, not typosquatting, avoiding double-flagging).

    Args:
        domain: Lowercased domain string.

    Returns:
        A flag message if likely typosquatting is detected, otherwise None.
    """
    labels = _get_comparable_labels(domain)
    if not labels:
        return None

    for label in labels:
        for brand in KNOWN_BRANDS:
            if label == brand:
                continue  # exact match after normalization = impersonation's job, not typosquat's

            similarity = _similarity(label, brand)
            if similarity >= 0.75:
                return (
                    f"Domain label '{label}' closely resembles known "
                    f"brand '{brand}' (similarity {similarity:.0%}) — possible "
                    f"typosquatting"
                )
    return None
import re
from urllib.parse import urlparse


def analyze_url(url: str) -> dict:
    """
    Analyze a single URL and return risk information.
    """

    score = 0
    flags = []

    url = url.strip().lower()

    # Missing HTTPS
    if not url.startswith("https://"):
        score += 15
        flags.append("URL does not use HTTPS")

# Extract domain
    try:
        parsed = urlparse(url)
        domain = parsed.netloc

        if not domain:
            domain = parsed.path.split("/")[0]

        # URL shortener detection
        if domain in URL_SHORTENERS:
            score += 30
            flags.append(
                "URL uses a shortening service (possible hidden destination)"
            )


    except Exception:
        return {
            "url": url,
            "risk_score": 100,
            "flags": ["Invalid URL"]
        }


    # Brand impersonation check
    brand_flag = _check_brand_impersonation(domain)

    if brand_flag:
        score += 40
        flags.append(brand_flag)


    # Typosquatting check
    typo_flag = _check_typosquatting(domain)

    if typo_flag:
        score += 35
        flags.append(typo_flag)


    # Suspicious keywords
    suspicious_words = [
    "login",
    "verify",
    "verification",
    "secure",
    "update",
    "account",
    "bank",
    "blocked",
    "suspended",
    "password",
    "confirm"
]
    for word in suspicious_words:
        if word in url:
            score += 5
            flags.append(
                f"Suspicious keyword detected: {word}"
            )


    score = min(score, 100)


    if score >= 80:
        level = "Critical Risk"
    elif score >= 60:
        level = "High Risk"
    elif score >= 30:
        level = "Suspicious"
    else:
        level = "Safe"


    return {
        "url": url,
        "risk_score": score,
        "risk_level": level,
        "flags": flags
    }



def analyze_urls(urls: list[str]) -> dict:
    """
    Analyze multiple URLs and return the exact structure required
    by risk_engine.py.
    """

    if not urls:
        return {
            "url_count": 0,
            "max_url_risk": 0,
            "max_phishing_ml_probability": None,
            "all_flags": [],
            "details": []
        }


    results = []

    max_risk = 0
    max_phishing_probability = None
    all_flags = []


    for url in urls:

        result = analyze_url(url)

        results.append(result)


        # Highest heuristic URL risk
        risk = result.get("risk_score", 0)

        if risk > max_risk:
            max_risk = risk


        # Collect flags
        all_flags.extend(
            result.get("flags", [])
        )


        # Future phishing ML compatibility
        phishing_probability = result.get(
            "phishing_ml_probability",
            None
        )

        if phishing_probability is not None:

            if (
                max_phishing_probability is None
                or phishing_probability > max_phishing_probability
            ):
                max_phishing_probability = phishing_probability



    return {
    "url_count": len(results),
    "max_url_risk": max_risk,
    "max_phishing_ml_probability": max_phishing_probability,
    "all_flags": all_flags,

    # compatibility with Streamlit UI
    "per_url_results": results,

    # keep old name also
    "details": results
}