"""
phishing_model.py

Trains phishing_classifier: a URL-based phishing detector, trained on the
offline-computable subset of the "Datasets for Phishing Websites Detection"
dataset (Vrbančič, Fister Jr., Podgorelec — Data in Brief, 2020). See
README.md for full dataset documentation and licensing.

Compares several algorithms head-to-head (Logistic Regression, Linear SVM,
Random Forest, Multinomial Naive Bayes, and XGBoost if installed), scores
each on accuracy/precision/recall/F1/ROC-AUC plus training and prediction
time, and automatically saves the best-performing one.

Each candidate is wrapped in a scikit-learn Pipeline (with StandardScaler
where the algorithm benefits from it) so the single saved artifact is
self-contained: url_checker.py can call .predict_proba() on a raw feature
row without needing to know which algorithm won or whether scaling is
required.
"""

import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import FunctionTransformer

from src.config import (
    PHISHING_CLASSIFIER_PATH,
    PHISHING_URL_PROCESSED_CSV,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.utils import ensure_file_exists, get_logger

logger = get_logger(__name__)

# Companion artifact: the exact ordered list of feature columns the model
# was trained on. url_checker.py must build its feature row in this exact
# order before calling predict_proba() — saved separately from the model
# itself so it's easy to inspect without unpickling the estimator.
PHISHING_FEATURE_COLUMNS_PATH = PHISHING_CLASSIFIER_PATH.parent / "phishing_feature_columns.joblib"

TARGET_COLUMN = "phishing"


def load_data() -> pd.DataFrame:
    """Load the offline-feature phishing dataset.

    Returns:
        DataFrame with 98 offline-computable URL/domain/directory/file/
        parameter features plus the "phishing" target column (0/1).
    """
    ensure_file_exists(
        PHISHING_URL_PROCESSED_CSV,
        "Processed phishing dataset (run the dataset preparation step first)",
    )
    df = pd.read_csv(PHISHING_URL_PROCESSED_CSV)
    logger.info("Loaded phishing dataset: %d rows, %d columns.", *df.shape)
    return df


def prepare_data(df: pd.DataFrame):
    """Split the dataset into stratified train/test feature matrices.

    Args:
        df: Full dataset including the target column.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_columns), where
        feature_columns is the ordered list of column names used — this is
        what gets saved alongside the model for inference-time alignment.
    """
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_columns


def _build_candidate_models() -> dict[str, Pipeline]:
    """Define the candidate models to compare, each as a full Pipeline.

    Returns:
        Dict of model_name -> unfitted sklearn Pipeline. Linear models get
        a StandardScaler step since they're sensitive to feature scale;
        tree-based and count-based models (Random Forest, Naive Bayes) do
        not, since scaling would either be pointless (trees) or invalid
        (Naive Bayes requires non-negative counts).
    """
    candidates: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            # LinearSVC has no predict_proba; CalibratedClassifierCV adds
            # probability estimates via cross-validated Platt scaling,
            # which risk_engine.py needs for a usable phishing_probability.
            ("model", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE, dual="auto"), cv=3)),
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "Multinomial Naive Bayes": Pipeline([
            # The source dataset uses -1 as a sentinel meaning "this URL
            # component (directory/file/params) is absent", which other
            # models handle fine as an ordinary numeric value, but
            # MultinomialNB requires strictly non-negative input. Clipping
            # to 0 here only affects this one candidate.
            ("clip_negative", FunctionTransformer(lambda X: X.clip(lower=0))),
            ("model", MultinomialNB()),
        ]),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = Pipeline([
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            )),
        ])
    except ImportError:
        logger.info("xgboost not installed — skipping XGBoost candidate (optional dependency).")

    return candidates


def train_and_evaluate(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Train one candidate pipeline and compute its evaluation metrics.

    Args:
        name: Human-readable model name, for logging/reporting.
        pipeline: Unfitted sklearn Pipeline.
        X_train, y_train: Training split.
        X_test, y_test: Held-out test split.

    Returns:
        Dict of metrics: name, pipeline (fitted), accuracy, precision,
        recall, f1_score, roc_auc, confusion_matrix, train_time_seconds,
        predict_time_seconds.
    """
    logger.info("Training %s...", name)

    train_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    predictions = pipeline.predict(X_test)
    predict_time = time.perf_counter() - predict_start

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "name": name,
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "train_time_seconds": round(train_time, 3),
        "predict_time_seconds": round(predict_time, 4),
    }

    logger.info(
        "%s -> accuracy=%.4f, f1=%.4f, roc_auc=%.4f, train_time=%.2fs",
        name, metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"], train_time,
    )

    return metrics


def compare_models(X_train, X_test, y_train, y_test) -> list[dict]:
    """Train and evaluate every candidate model.

    Args:
        X_train, X_test, y_train, y_test: Train/test split from prepare_data().

    Returns:
        List of metrics dicts (see train_and_evaluate), one per candidate,
        in the order they were trained.
    """
    candidates = _build_candidate_models()
    results = []

    for name, pipeline in candidates.items():
        result = train_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(result)

    return results


def print_comparison_report(results: list[dict]) -> None:
    """Print a formatted comparison table of all candidate models.

    Args:
        results: Output of compare_models().
    """
    print("\n========== PHISHING MODEL COMPARISON ==========\n")
    header = f"{'Model':<26}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>9}{'Train(s)':>10}{'Predict(s)':>12}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['name']:<26}"
            f"{result['accuracy']:>10.4f}"
            f"{result['precision']:>11.4f}"
            f"{result['recall']:>9.4f}"
            f"{result['f1_score']:>8.4f}"
            f"{result['roc_auc']:>9.4f}"
            f"{result['train_time_seconds']:>10.2f}"
            f"{result['predict_time_seconds']:>12.4f}"
        )
    print()


def select_best_model(results: list[dict]) -> dict:
    """Select the best-performing model by F1 score.

    F1 is used as the primary criterion (rather than raw accuracy) because
    it balances precision and recall — for a phishing detector, both
    false positives (flagging legitimate mail) and false negatives
    (missing real phishing) carry real cost.

    Args:
        results: Output of compare_models().

    Returns:
        The single best result dict.
    """
    best = max(results, key=lambda result: result["f1_score"])
    logger.info("Best model selected: %s (F1=%.4f)", best["name"], best["f1_score"])
    return best


def save_model(best_result: dict, feature_columns: list[str]) -> None:
    """Persist the winning pipeline and its feature column order.

    Args:
        best_result: The result dict returned by select_best_model().
        feature_columns: Ordered feature column names used during training.
    """
    joblib.dump(best_result["pipeline"], PHISHING_CLASSIFIER_PATH)
    joblib.dump(feature_columns, PHISHING_FEATURE_COLUMNS_PATH)
    logger.info("Saved phishing_classifier to %s", PHISHING_CLASSIFIER_PATH)
    logger.info("Saved feature column order to %s", PHISHING_FEATURE_COLUMNS_PATH)


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, feature_columns = prepare_data(df)

    print(f"Training rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]} | Features: {len(feature_columns)}")

    results = compare_models(X_train, X_test, y_train, y_test)
    print_comparison_report(results)

    best_result = select_best_model(results)

    print(f"\nBest model: {best_result['name']}")
    print("\nConfusion Matrix:")
    print(best_result["confusion_matrix"])

    save_model(best_result, feature_columns)
    print(f"\nphishing_classifier saved successfully ({best_result['name']}).")
