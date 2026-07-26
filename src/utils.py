"""
utils.py

Shared, reusable helpers used across the project: logging setup and small
utility functions that don't belong to any single pipeline stage.

Nothing in preprocess.py, feature_engineering.py, train.py, etc. should
configure logging on its own — they all call get_logger() from here so
every module logs in the same format to the same file.
"""

import logging
from pathlib import Path
from typing import Any

from src.config import LOG_DATE_FORMAT, LOG_FILE, LOG_FORMAT


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a configured logger.

    Safe to call multiple times with the same name: logging.getLogger()
    returns the same underlying Logger instance, and the handler-count
    check below prevents duplicate log lines if a module is re-imported.

    Args:
        name: Logger name, conventionally the calling module's __name__.
        level: Minimum severity this logger will emit.

    Returns:
        A configured logging.Logger instance that writes to both the
        console and LOG_FILE.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Already configured (e.g. Streamlit re-running the script on
        # every interaction) — don't attach duplicate handlers.
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def ensure_file_exists(path: Path, description: str = "File") -> None:
    """Raise a clear, actionable error if a required file is missing.

    Used to fail fast with a helpful message (e.g. "Model file not found —
    did you run train.py?") instead of letting a bare FileNotFoundError
    from deep inside joblib/pandas confuse the user.

    Args:
        path: Path that must exist.
        description: Human-readable label used in the error message.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at: {path}. "
            f"Check that the required pipeline step has been run."
        )


def safe_get(dictionary: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a value from a dict without raising on a missing or None key.

    A tiny convenience wrapper used when reading optional fields out of
    parsed .eml messages or uploaded CSV rows, where a key may be absent
    entirely or present with a None value.

    Args:
        dictionary: The dict to read from.
        key: The key to look up.
        default: Value to return if key is missing or maps to None.

    Returns:
        dictionary[key] if present and not None, otherwise default.
    """
    value = dictionary.get(key, default)
    return default if value is None else value


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text for safe display (e.g. in tables or logs).

    Args:
        text: The text to truncate.
        max_length: Maximum number of characters to keep.

    Returns:
        The original text if short enough, otherwise the first
        max_length characters followed by an ellipsis.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."
