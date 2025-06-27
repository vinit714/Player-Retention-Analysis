import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # For saving and loading models
import streamlit as st

def split_data(df: pd.DataFrame, target_col='churn'):
    """
    Split dataset into features and target, then train-test split.
    Args:
        df (pd.DataFrame): Processed data.
        target_col (str): Target column name.
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=['player_id', target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_random_forest(X_train, y_train, use_grid_search=False):
    """
    Train Random Forest model, optionally with Grid Search hyperparameter tuning.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        use_grid_search (bool): Whether to perform hyperparameter tuning.
    Returns:
        model: Trained Random Forest model.
    """
    if use_grid_search:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        st.write(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        return rf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data and print classification report & confusion matrix.
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, filepath: str):
    """
    Save the trained model to disk.
    Args:
        model: Trained model.
        filepath (str): File path to save the model.
    """
    joblib.dump(model, filepath)

def load_model(filepath: str):
    """
    Load a saved model from disk.
    Args:
        filepath (str): File path of saved model.
    Returns:
        Loaded model.
    """
    return joblib.load(filepath)

def train_and_evaluate(df: pd.DataFrame, use_grid_search=False, model_path='models/rf_model.joblib'):
    """
    Full pipeline to train and evaluate the model, and save it.
    Args:
        df (pd.DataFrame): Processed data.
        use_grid_search (bool): Whether to tune hyperparameters.
        model_path (str): Path to save trained model.
    Returns:
        model, X_test (for further use)
    """
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_random_forest(X_train, y_train, use_grid_search)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_path)
    return model, X_test
