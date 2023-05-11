import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
# from typing import Tuple
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas dataframe.
    """
    data = pd.read_csv(file_path)
    return data

def split_features_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split data into features and labels.
    """
    x = data.drop('label', axis=1)
    y = data['label']
    return x, y


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing values in a pandas dataframe with the mean of the column.
    """
    print(data.isna().sum())
    missing_values = data.isna().sum()
    if missing_values.any():
        data.fillna(data.mean(), inplace=True)
    else:
        print("\nMissing values not found")
    return data


def scale_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the features in a pandas dataframe to have zero mean and unit variance.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale the features in train and test dataframes to have zero mean and unit variance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data features
    X_test : pd.DataFrame
        Test data features

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Scaled training and test data features
    """
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test


def encode_labels(labels: pd.Series) -> pd.Series:
    """
    Encode categorical labels in a pandas dataframe as numerical values.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels


def split_data(data: pd.DataFrame, labels: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split the input data and labels into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data.
    """
    data = handle_missing_values(data)
    data = scale_features(data)
    return data


def split_data_train_val_test(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split the input data into training, validation, and testing sets.
    """
    X_train_val, X_test, y_train_val, y_test = split_data(data.iloc[:, :-1], data.iloc[:, -1], test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = split_data(X_train_val, y_train_val, test_size=test_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data(data, file_path: str):
    """
    Save data to a file.
    """
    joblib.dump(data, file_path)
