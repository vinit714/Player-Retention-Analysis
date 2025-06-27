import pandas as pd
import numpy as np

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw player data CSV file.
    Args:
        filepath (str): Path to raw CSV data.
    Returns:
        pd.DataFrame: Raw data loaded as DataFrame.
    """
    data = pd.read_csv(filepath)
    return data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by handling missing values and duplicates.
    Args:
        df (pd.DataFrame): Raw data.
    Returns:
        pd.DataFrame: Cleaned data.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features like avg session time, encode categorical variables.
    Args:
        df (pd.DataFrame): Cleaned data.
    Returns:
        pd.DataFrame: Data with engineered features.
    """
    # Average session length
    df['session_avg_time'] = df['total_playtime'] / df['num_sessions']

    # One-hot encode categorical columns
    categorical_cols = ['country', 'device']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

def save_processed_data(df: pd.DataFrame, filepath: str):
    """
    Save processed DataFrame to CSV.
    Args:
        df (pd.DataFrame): Processed data.
        filepath (str): Path to save CSV.
    """
    df.to_csv(filepath, index=False)

def load_and_process(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Full pipeline to load, clean, engineer, and save data.
    Args:
        raw_path (str): Path to raw CSV.
        processed_path (str): Path to save processed CSV.
    Returns:
        pd.DataFrame: Processed data ready for modeling.
    """
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)
    save_processed_data(df, processed_path)
    return df

def show_eda(df: pd.DataFrame):
    """
    Simple EDA prints and plots (to be used in notebooks or Streamlit).
    """
    print("Data sample:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nData description:")
    print(df.describe())
