import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def load_data():
    """
    Load the cleaned dataset.
    """
    df = pd.read_csv("data/processed/cleaned_spam.csv")

    # Handle any empty messages
    df["message"] = df["message"].fillna("empty")

    return df


def prepare_data(df):
    """
    Convert text into TF-IDF features.
    """
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    return X, y, vectorizer


def train_model(X_train, y_train):
    """
    Train the Naive Bayes model.
    """
    model = MultinomialNB()

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """
    predictions = model.predict(X_test)

    print("\n========== MODEL EVALUATION ==========\n")

    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, pos_label='spam'):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, pos_label='spam'):.4f}")
    print(f"F1 Score : {f1_score(y_test, predictions, pos_label='spam'):.4f}")

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))


def save_model(model, vectorizer):
    """
    Save model and vectorizer.
    """
    joblib.dump(model, "models/spam_classifier.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("\nModel saved successfully.")


if __name__ == "__main__":

    # Load dataset
    df = load_data()

    # Convert text to features
    X, y, vectorizer = prepare_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save
    save_model(model, vectorizer)