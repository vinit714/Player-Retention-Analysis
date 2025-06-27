import pytest
import pandas as pd
from src import model

def get_sample_data():
    df = pd.DataFrame({
        'player_id': [1, 2, 3, 4],
        'num_sessions': [5, 10, 3, 8],
        'total_playtime': [50, 200, 30, 100],
        'in_game_purchases': [1, 3, 0, 2],
        'days_since_last_session': [2, 0, 15, 1],
        'level_reached': [10, 20, 5, 15],
        'session_avg_time': [10, 20, 10, 12.5],
        'achievements_unlocked': [5, 10, 2, 8],
        'social_interactions': [3, 5, 0, 4],
        'country_USA': [0, 1, 0, 0],
        'device_Mobile': [1, 0, 0, 1],
        'churn': [0, 0, 1, 0]
    })
    return df

def test_split_data():
    df = get_sample_data()
    X_train, X_test, y_train, y_test = model.split_data(df)
    assert len(X_train) > 0 and len(X_test) > 0

def test_train_random_forest():
    df = get_sample_data()
    X_train, X_test, y_train, y_test = model.split_data(df)
    rf_model = model.train_random_forest(X_train, y_train)
    preds = rf_model.predict(X_test)
    assert len(preds) == len(y_test)
