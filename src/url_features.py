"""
url_features.py

Extracts URLs from raw email text and computes objective, measurable
features for each one (length, character composition, entropy, TLD,
IP-literal detection, etc.).

This module deliberately contains NO judgment calls ("is this suspicious?").
It only measures. Judgment (is this TLD risky? is this a known shortener?
does this look like typosquatting?) belongs in url_checker.py, which
consumes the output of this module. Keeping the two separated means new
reputation rules can be added later without touching feature extraction.

IMPORTANT: URL extraction must run on the RAW email body, before
preprocess_text() strips punctuation and lowercases everything — otherwise
"https://amazon-security-login.xyz" becomes unrecognizable as a URL.
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# URL detection
# --------------------------------------------------------------------------
# Matches http(s):// URLs and bare www.-prefixed hosts. Stops at whitespace
# or common trailing punctuation/wrapper characters that are unlikely to be
# part of the URL itself (quotes, closing parens, sentence punctuation).
_URL_PATTERN = re.compile(
    r"""(?:https?://|www\.)[^\s<>"'\)\]\}]+""",
    re.IGNORECASE,
)

# Trailing characters that are almost always punctuation, not part of the
# URL, when they appear at the very end of a match (e.g. a URL followed by
# a period at the end of a sentence).
_TRAILING_PUNCTUATION = ".,;:!?'\""


def extract_urls(email_body: str) -> list[str]:
    """Extract all URLs found in raw (unprocessed) email text.

    Args:
        email_body: Raw email body text, BEFORE any text preprocessing.

    Returns:
        List of URL strings (url_list) in the order they appear. Bare
        "www."-prefixed URLs are normalized to include an "http://" scheme
        so downstream parsing with urlparse() works consistently.
    """
    if not email_body:
        return []

    matches = _URL_PATTERN.findall(email_body)

    url_list: list[str] = []
    for raw_url in matches:
        cleaned = raw_url.rstrip(_TRAILING_PUNCTUATION)

        if not cleaned:
            continue

        if cleaned.lower().startswith("www."):
            cleaned = "http://" + cleaned

        url_list.append(cleaned)

    return url_list


def _get_domain(url: str) -> str:
    """Extract the network location (host[:port]) from a URL, minus port.

    Args:
        url: A full URL string.

    Returns:
        Lowercased hostname, e.g. "amazon-security-login.xyz". Empty
        string if the URL cannot be parsed.
    """
    try:
        netloc = urlparse(url).netloc
    except ValueError:
        return ""

    # Strip credentials (user:pass@) and port if present.
    netloc = netloc.split("@")[-1]
    netloc = netloc.split(":")[0]

    return netloc.lower()


def _shannon_entropy(text: str) -> float:
    """Compute the Shannon entropy of a string, in bits per character.

    Randomly generated domains (common in automated phishing kits) tend to
    have higher entropy than human-chosen brand-like domains. This is a
    measurement, not a verdict — url_checker.py decides what entropy level
    counts as suspicious.

    Args:
        text: The string to measure (typically a domain name).

    Returns:
        Entropy in bits per character. 0.0 for empty or single-repeated-
        character strings.
    """
    if not text:
        return 0.0

    length = len(text)
    counts = Counter(text)

    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return round(entropy, 4)


def _is_ip_address(host: str) -> bool:
    """Check whether a host string is a literal IPv4 address.

    Args:
        host: Hostname portion of a URL (no port).

    Returns:
        True if host is a dotted-quad IPv4 address.
    """
    if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        return False
    return all(0 <= int(octet) <= 255 for octet in host.split("."))


def extract_url_features(url: str) -> dict:
    """Compute the full objective feature set for a single URL.

    Args:
        url: A URL string, as returned by extract_urls().

    Returns:
        A dict of features (this becomes one row of feature_matrix when
        multiple URLs are analyzed together). Keys:
            url                 - the original URL string
            domain              - hostname portion
            path                - URL path
            query               - URL query string
            url_length          - total character length of the URL
            domain_length       - character length of the domain
            uses_https          - bool, scheme is https
            has_ip_address      - bool, host is a literal IP
            num_dots            - dot count in the domain
            num_hyphens         - hyphen count in the domain
            num_digits          - digit count in the domain
            num_subdomains      - count of subdomain labels before the
                                   registrable domain (e.g. "a.b.example.com"
                                   -> 2)
            tld                 - top-level domain label (e.g. "xyz")
            has_at_symbol       - bool, "@" present anywhere in the URL
                                   (classic credential-hiding trick)
            domain_entropy      - Shannon entropy of the domain string
            path_length         - character length of the path
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        parsed = None

    domain = _get_domain(url)
    scheme = parsed.scheme.lower() if parsed else ""
    path = parsed.path if parsed else ""
    query = parsed.query if parsed else ""

    domain_labels = [label for label in domain.split(".") if label]
    num_subdomains = max(0, len(domain_labels) - 2)
    tld = domain_labels[-1] if domain_labels else ""

    return {
        "url": url,
        "domain": domain,
        "path": path,
        "query": query,
        "url_length": len(url),
        "domain_length": len(domain),
        "uses_https": scheme == "https",
        "has_ip_address": _is_ip_address(domain),
        "num_dots": domain.count("."),
        "num_hyphens": domain.count("-"),
        "num_digits": sum(character.isdigit() for character in domain),
        "num_subdomains": num_subdomains,
        "tld": tld,
        "has_at_symbol": "@" in url,
        "domain_entropy": _shannon_entropy(domain.replace(".", "")),
        "path_length": len(path),
    }


def extract_features_for_urls(url_list: list[str]) -> list[dict]:
    """Compute features for every URL in a list.

    Args:
        url_list: URLs as returned by extract_urls().

    Returns:
        feature_matrix — a list of per-URL feature dicts, in the same
        order as url_list.
    """
    return [extract_url_features(url) for url in url_list]


# --------------------------------------------------------------------------
# Full offline feature vector (for phishing_classifier)
# --------------------------------------------------------------------------
# These feature names and definitions mirror the 98 offline-computable
# columns of the "Datasets for Phishing Websites Detection" dataset
# (Vrbančič, Fister Jr., Podgorelec — Data in Brief, 2020), MINUS the ~13
# columns that require a live DNS/WHOIS/TLS/Google-index lookup. Matching
# these exact column names/order is required so phishing_model.py (trained
# on that dataset) and this function (used at prediction time) speak the
# same feature language. See README.md "Datasets" section for the full
# citation and the network-dependent columns that were excluded.

# (feature_name_suffix, literal_character) pairs counted in each URL
# component. Defined once and reused across url/domain/directory/file/
# params instead of writing five nearly-identical blocks of code.
_SPECIAL_CHARACTERS: tuple[tuple[str, str], ...] = (
    ("dot", "."), ("hyphen", "-"), ("underline", "_"), ("slash", "/"),
    ("questionmark", "?"), ("equal", "="), ("at", "@"), ("and", "&"),
    ("exclamation", "!"), ("space", " "), ("tilde", "~"), ("comma", ","),
    ("plus", "+"), ("asterisk", "*"), ("hashtag", "#"), ("dollar", "$"),
    ("percent", "%"),
)

_VOWELS = frozenset("aeiouAEIOU")

_SERVER_CLIENT_KEYWORDS = ("server", "client")

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Known URL-shortening services. Defined here (not in url_checker.py) so
# both feature extraction (the url_shortened flag, used by the ML model)
# and reputation checking (url_checker.py's flag/score logic) share a
# single source of truth instead of maintaining two copies of this list.
KNOWN_SHORTENER_DOMAINS: frozenset[str] = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "rebrand.ly", "cutt.ly", "rb.gy", "tiny.cc",
    "s.id", "v.gd",
})


def _count_special_characters(text: str, component: str) -> dict:
    """Count each special character in a URL component.

    Args:
        text: The substring to count characters in (e.g. the domain).
        component: Suffix used in the output key names, e.g. "domain"
            produces keys like "qty_dot_domain".

    Returns:
        Dict of qty_<char>_<component> -> count, one entry per character
        in _SPECIAL_CHARACTERS.
    """
    return {
        f"qty_{name}_{component}": text.count(character)
        for name, character in _SPECIAL_CHARACTERS
    }


def _split_path_into_directory_and_file(path: str) -> tuple[str, str]:
    """Split a URL path into its directory and file components.

    Approximates the convention used by the source dataset: the final
    path segment is treated as a "file" if it looks like a filename
    (contains a dot and isn't just a trailing slash); everything before
    it is the "directory". This is a best-effort reconstruction since the
    original feature-extraction source code is not published — training
    and inference both use this same logic, which is what matters for
    model consistency.

    Args:
        path: The URL path component (e.g. "/account/verify.php").

    Returns:
        (directory, file) tuple of strings.
    """
    if not path or path == "/":
        return "", ""

    segments = path.strip("/").split("/")
    last_segment = segments[-1]

    looks_like_file = "." in last_segment and not path.endswith("/")

    if looks_like_file and len(segments) >= 1:
        directory = "/" + "/".join(segments[:-1]) if len(segments) > 1 else ""
        file_part = last_segment
    else:
        directory = path
        file_part = ""

    return directory, file_part


def extract_full_feature_vector(url: str) -> dict:
    """Compute the full offline feature vector used by phishing_classifier.

    This produces one row matching the training columns of
    data/processed/cleaned_phishing_urls.csv exactly (column names and
    all), so the trained model can be applied directly to the output of
    this function without any remapping.

    Args:
        url: A URL string, as returned by extract_urls().

    Returns:
        Dict of 98 feature_name -> value, ready to be assembled into a
        single-row feature_matrix for phishing_classifier.predict().
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        parsed = None

    domain = _get_domain(url)
    path = parsed.path if parsed else ""
    query = parsed.query if parsed else ""
    directory, file_part = _split_path_into_directory_and_file(path)

    features: dict = {}

    # --- URL-level counts ---
    features.update(_count_special_characters(url, "url"))
    domain_labels = [label for label in domain.split(".") if label]
    tld = domain_labels[-1] if domain_labels else ""
    features["qty_tld_url"] = len(tld)
    features["length_url"] = len(url)

    # --- Domain-level counts ---
    features.update(_count_special_characters(domain, "domain"))
    features["qty_vowels_domain"] = sum(1 for character in domain if character in _VOWELS)
    features["domain_length"] = len(domain)
    features["domain_in_ip"] = int(_is_ip_address(domain))
    features["server_client_domain"] = int(
        any(keyword in domain.lower() for keyword in _SERVER_CLIENT_KEYWORDS)
    )

    # --- Directory-level counts ---
    # The training dataset uses -1 (not 0) as a sentinel meaning "this URL
    # has no directory component at all" — distinct from a directory that
    # exists but is empty. The vast majority of ordinary URLs (a bare
    # domain, or domain + single file with no subpath) have NO directory,
    # so this distinction matters enormously for prediction accuracy: an
    # inference-time 0 here would put ~60% of legitimate URLs far outside
    # the distribution the model was trained on.
    if directory:
        features.update(_count_special_characters(directory, "directory"))
        features["directory_length"] = len(directory)
    else:
        for name, _ in _SPECIAL_CHARACTERS:
            features[f"qty_{name}_directory"] = -1
        features["directory_length"] = -1

    # --- File-level counts ---
    if file_part:
        features.update(_count_special_characters(file_part, "file"))
        features["file_length"] = len(file_part)
    else:
        for name, _ in _SPECIAL_CHARACTERS:
            features[f"qty_{name}_file"] = -1
        features["file_length"] = -1

    # --- Parameter-level counts ---
    if query:
        features.update(_count_special_characters(query, "params"))
        features["params_length"] = len(query)
        features["tld_present_params"] = int(bool(tld) and tld.lower() in query.lower())
        features["qty_params"] = query.count("=")
    else:
        for name, _ in _SPECIAL_CHARACTERS:
            features[f"qty_{name}_params"] = -1
        features["params_length"] = -1
        features["tld_present_params"] = -1
        features["qty_params"] = -1

    # --- Whole-URL semantic flags ---
    features["email_in_url"] = int(bool(_EMAIL_PATTERN.search(url)))
    features["url_shortened"] = int(domain in KNOWN_SHORTENER_DOMAINS)

    return features


def extract_full_feature_matrix(url_list: list[str]) -> list[dict]:
    """Compute the full offline feature vector for every URL in a list.

    Args:
        url_list: URLs as returned by extract_urls().

    Returns:
        List of dicts from extract_full_feature_vector(), one per URL, in
        the same order as url_list.
    """
    return [extract_full_feature_vector(url) for url in url_list]
