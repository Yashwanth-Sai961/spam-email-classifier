import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    df = pd.read_csv("data/processed/cleaned_spam.csv")

    print("Before fixing:")
    print(df.isnull().sum())

    df["message"] = df["message"].fillna("empty")

    print("\nAfter fixing:")
    print(df.isnull().sum())

    return df

def count_vectorizer(df):
    """
    Convert text into numerical features using CountVectorizer.
    """
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    print("\n===== CountVectorizer =====")
    print("Feature Matrix Shape:", X.shape)

    # Save vectorizer
    joblib.dump(vectorizer, "models/count_vectorizer.joblib")

    return X, y


def tfidf_vectorizer(df):
    """
    Convert text into numerical features using TF-IDF.
    """
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    print("\n===== TF-IDF Vectorizer =====")
    print("Feature Matrix Shape:", X.shape)

    # Save vectorizer
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    return X, y


if __name__ == "__main__":

    df = load_data()

    X_count, y_count = count_vectorizer(df)

    X_tfidf, y_tfidf = tfidf_vectorizer(df)

    print("\nVectorizers saved successfully!")