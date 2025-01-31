import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset (e.g., missing values, encoding)."""
    df = df.dropna()  # Example preprocessing
    return df