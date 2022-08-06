"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""
import pandas as pd
import numpy as np
from pyrsistent import l


def mushrooms_raw(mushrooms: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for mushroom dataset.

    Args:
        mushrooms: Raw data.
    Returns:
        raw data stored as csv file
    """
    return mushrooms

def preprocess_mushrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for mushroom dataset.

    Args:
        mushrooms: Raw data.
    Returns:
        Preprocessed data stored as csv file (removes missing values)
    """
    # stalk-root has missing values
    # Either remove the ? or remove the column entirely
    # df = df[df['stalk-root'] != '?']
    df = df.drop(columns='stalk-root')
    
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for mushroom dataset.

    Args:
        mushrooms: Parquet.
    Returns:
        Preprocessed data stored as parquet file
    """
    labels = ['edible']

    #Identify all columns to use dummy variables (Except for label)
    dummy_cols = list(set(df.columns) - set(labels)) 
    df = pd.get_dummies(df, prefix_sep = '_', columns=dummy_cols)
    df[labels] = np.where(df[labels] == 'e', 1, 0) 
    return df

def documentation(documentation: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        documentation: .name file for dataset documentation.
    Returns:
        documentation (To be published as a markdown .md file)
    """
    return documentation