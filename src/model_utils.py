import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def get_processed_data(data_location: str) -> pd.DataFrame:
    """
    Load processed data from a specified location.

    Args:
        data_location (str): Location to get the data.

    Returns:
        pd.DataFrame: Processed data.
    """
    return joblib.load(data_location)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """
    Train the model using grid search cross-validation.
    """
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    return grid


def predict(grid: GridSearchCV, X_test: pd.DataFrame) -> pd.Series:
    """
    Use the trained model to make predictions on test data.
    """
    predictions = grid.predict(X_test)
    return predictions


def save_model(model: GridSearchCV, save_path: str):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, save_path)


def save_predictions(predictions: pd.Series, save_path: str):
    """
    Save the predictions to a file.
    """
    predictions.to_csv(save_path, index=False)