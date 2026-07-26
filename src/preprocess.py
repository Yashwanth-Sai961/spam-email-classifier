import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only needed the first time)
nltk.download("stopwords")

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def load_data():
    """
    Load the raw SMS Spam Collection dataset.
    """
    df = pd.read_csv(
        "data/raw/spam.csv",
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )

    return df


def clean_dataset(df):
    """
    Remove duplicate rows and reset the index.
    """
    print(f"Rows before removing duplicates: {df.shape[0]}")

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Rows after removing duplicates: {df.shape[0]}")

    return df


def preprocess_text(text):
    """
    Clean a single SMS message.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Apply stemming
    words = [
        stemmer.stem(word)
        for word in words
    ]

    cleaned_text = " ".join(words)

    # Prevent empty messages
    if cleaned_text.strip() == "":
        return "empty"

    return cleaned_text


def preprocess_dataset(df):
    """
    Apply preprocessing to every message.
    """
    df["message"] = df["message"].apply(preprocess_text)

    return df


def save_dataset(df):
    """
    Save the cleaned dataset.
    """
    df.to_csv(
        "data/processed/cleaned_spam.csv",
        index=False
    )

    print("\nCleaned dataset saved successfully!")


if __name__ == "__main__":

    # Load data
    df = load_data()

    # Remove duplicates
    df = clean_dataset(df)

    # Clean all messages
    df = preprocess_dataset(df)

    # Display sample
    print("\nFirst 5 cleaned messages:\n")
    print(df.head())

    # Save processed dataset
    save_dataset(df)