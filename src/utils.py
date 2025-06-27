import pandas as pd

def save_df_to_csv(df: pd.DataFrame, filepath: str):
    """
    Save DataFrame to CSV with index=False.
    """
    df.to_csv(filepath, index=False)

def load_df_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV.
    """
    return pd.read_csv(filepath)

def check_file_exists(filepath: str) -> bool:
    """
    Check if a file exists.
    """
    import os
    return os.path.exists(filepath)
