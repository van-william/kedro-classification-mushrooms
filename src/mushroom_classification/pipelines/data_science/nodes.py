"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""


import logging
from os import access
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt

import xgboost as xgb
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
    """Calculates and logs the F1 Score.

    Args:
        classifier: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for edible (Classifier).
    """
    y_pred = classifier.predict(X_test)
    clf_f1 = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has an F1 score of %.3f on test data.", clf_f1)


def iterative_training(df:pd.DataFrame, feature_imp:pd.DataFrame, parameters: Dict):
    """Calculates the important features with relative importance values

    Args:
        df: preprocessed_mushrooms
        feature_imp: dataframe of sorted importance features (base features)
        parameters: 
            feature_count: number of paramters
    """
    accuracy_list = []
    feature_name_list = []
    feature_count = list(range(1, parameters['feature_count']+1))
    for features in feature_count:
        feature_list = list(feature_imp.base_features[0:features])
        print(feature_list)
        labels = ['edible']
        #Identify all columns to use dummy variables (Except for label)
        df[labels] = np.where(df[labels] == 'e', 1, 0) 
        X = df[feature_list]
        X = pd.get_dummies(X, prefix_sep = '_')
        xgb_model = xgb.XGBClassifier()
        y = df[labels]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'])
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        predictions_xgb = [round(value) for value in y_pred_xgb]
        accuracy = accuracy_score(y_test, predictions_xgb)
        logger = logging.getLogger(__name__)
        logger.info(f"With {features} features used, model has accuracy of: {accuracy}")
        accuracy_list.append(accuracy)
        feature_name_list.append(' | '.join(map(str,feature_list)))

    summary_results = pd.DataFrame({"feature_count": feature_count, "accuracy": accuracy_list, "features": feature_name_list})
    return summary_results


def simple_decision_tree(df:pd.DataFrame, feature_imp:pd.DataFrame, parameters: Dict):
    feature_count = parameters['tree_feature_count']+1
    feature_list = list(feature_imp.base_features[0:feature_count])
    labels = ['edible']
    #Identify all columns to use dummy variables (Except for label)
    df[labels] = np.where(df[labels] == 'e', 1, 0) 
    X = df[feature_list]
    X = pd.get_dummies(X, prefix_sep = '_')
    y = df[labels]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'])

    tree_clf = tree.DecisionTreeClassifier(max_depth=parameters['max_depth'])
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger = logging.getLogger(__name__)
    logger.info(f"With {feature_count} features used, decision tree model has accuracy of: {accuracy}")
    logger.info(f"Size of X_train: {X_train.shape}")
    logger.info(f"X_train positive (edible): {y_train.sum()}")
    # example output file
    # plt.savefig('./data/08_reporting/'+image_name)
    tree_name = parameters['tree_name']
    #dot_data = tree.export_graphviz(tree_clf, out_file='data/08_reporting/'+tree_name, 
    feature_names=X.columns
    class_names = ['NOT edible', 'edible']
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    tree.plot_tree(tree_clf,
               feature_names = feature_names, 
               class_names=class_names,
               filled = True)
    fig.savefig('data/08_reporting/'+tree_name)

    return tree_clf
