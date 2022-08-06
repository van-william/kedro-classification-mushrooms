"""
This is a boilerplate pipeline 'exploratory_data_analysis'
generated using Kedro 0.18.2
"""

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

def pca_analysis(df:pd.DataFrame, parameters: Dict):
    """Calculates and logs the coefficient of determination.

    Args:
        df: dataframe of normalized data
        parameters: clarification of number of PCA components and file locations
    """
    plt.close('all')

    pca = PCA(n_components=parameters['n_components'])
    pca_df = pd.DataFrame(data = pca.fit_transform(df), columns = ['PC2_1', 'PC2_2'])
    combined_df = pd.concat([df['edible'], pca_df], axis=1, join='inner')
    scatter_plot = sns.scatterplot(x='PC2_1', y='PC2_2',hue='edible', data=combined_df)
    image_name = parameters['image_name']
    plt.savefig('./data/08_reporting/'+image_name, pad_inches=0.3)
    scatter_plot.clear()
    plt.close('all')
    return None

def corr_heat_map(df:pd.DataFrame, parameters: Dict):
    """Calculates and logs the coefficient of determination.

    Args:
        df: dataframe of normalized data
        parameters: file locations
    """
    # Custom Heatmap Plot",
    # Source: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec\r\n",
    plt.close('all')

    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 15))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 11, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    fig = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
    square=True, linewidths=.5, cbar_kws={"shrink": .5})

    image_name = parameters['image_name']
    
    plt.savefig('./data/08_reporting/'+image_name, pad_inches=0.5)
    fig.clear()
    plt.close('all')
    return None


def initial_feature_importance(df:pd.DataFrame, parameters: Dict):
    """Calculates the important features with relative importance values

    Args:
        df: normalized dataframe
        parameters: 
            test_size
    """
    xgb_model = xgb.XGBClassifier()
    X = df.drop(columns = 'edible')
    y = df.edible
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'])
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    predictions_xgb = [round(value) for value in y_pred_xgb]
    accuracy = np.mean(predictions_xgb == y_test)
    ### Identify Critical Features
    importance = (xgb_model.feature_importances_)
    features = X.columns
    importance_df = pd.DataFrame({"features": features, "importances": importance})
    importance_df = importance_df.sort_values(by=['importances'], ascending=False)
    logger = logging.getLogger(__name__)
    top_5 = importance_df.head(5)['features']
    logger.info(f"Model's top 5 features: {top_5}")
    
    importance_df['base_features'] = importance_df['features'].map(lambda x: x.split('_')[0])
    base_importance = importance_df.groupby(['base_features'])['importances'].sum()
    base_importance = base_importance.sort_values(ascending=False)

    base_importance_df = pd.DataFrame({"base_features": base_importance.index, "importances": base_importance})

    top_5_base = base_importance_df.head(5)
    logger.info(f"Model's top 5 base features: {top_5_base}")
    return importance_df, base_importance_df






