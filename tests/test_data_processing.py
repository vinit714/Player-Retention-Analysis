import pytest
import pandas as pd
from src import data_processing

def test_load_raw_data():
    df = data_processing.load_raw_data('data/raw/player_data_enhanced.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_clean_data():
    raw_df = pd.DataFrame({
        'player_id': [1, 2, 2, 3],
        'total_playtime': [10, 20, 20, None],
        'num_sessions': [5, 10, 10, 3]
    })
    cleaned = data_processing.clean_data(raw_df)
    # duplicates removed, NaN dropped
    assert cleaned.shape[0] == 2

def test_feature_engineering():
    df = pd.DataFrame({
        'total_playtime': [10, 20],
        'num_sessions': [5, 10],
        'country': ['India', 'USA'],
        'device': ['PC', 'Mobile']
    })
    engineered = data_processing.feature_engineering(df)
    assert 'session_avg_time' in engineered.columns
    assert 'country_USA' in engineered.columns
    assert 'device_Mobile' in engineered.columns
