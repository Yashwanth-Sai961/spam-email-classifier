import pandas as pd


def load_data():
    """
    Load the spam dataset and return a DataFrame.
    """
    df = pd.read_csv(
    "data/raw/spam.csv",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

    # Keep only the first two columns
    df = df.iloc[:, :2]

    # Rename columns
    df.columns = ["label", "message"]

    return df


def explore_data(df):
    print("\n========== DATASET INFORMATION ==========")
    print(f"Shape: {df.shape}")

    print("\nColumn Names:")
    print(df.columns)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nClass Distribution:")
    print(df["label"].value_counts())
    print("\nMessage Length Statistics:")
    print(df["message"].str.len().describe())

    print("\nLongest Message:")
    print(df.loc[df["message"].str.len().idxmax(), "message"])

    print("\nShortest Message:")
    print(df.loc[df["message"].str.len().idxmin(), "message"])
        


if __name__ == "__main__":
    dataframe = load_data()
    explore_data(dataframe)