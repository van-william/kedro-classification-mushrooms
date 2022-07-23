"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""


import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
#import plotly.express as px
#from kedro.extras.datasets.plotly import JSONDataSet


def split_data(df: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = df.drop(columns="edible")
    y = df["edible"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_model(
    classifier: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = classifier.predict(X_test)
    f1_score = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has an F1 score of %.3f on test data.", f1_score)
