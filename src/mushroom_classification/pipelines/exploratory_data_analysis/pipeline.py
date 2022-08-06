"""
This is a boilerplate pipeline 'exploratory_data_analysis'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import pca_analysis, corr_heat_map, initial_feature_importance

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=pca_analysis,
                inputs=["normalized_mushrooms", "params:pca_analysis"],
                outputs=None,
                name="PCA_analysis",
            ),
            node(
                func=corr_heat_map,
                inputs=["normalized_mushrooms", "params:corr_heat_map"],
                outputs=None,
                name="Heatmap_Correlation",
            ),
            node(
                func=initial_feature_importance,
                inputs=["normalized_mushrooms", "params:initial_feature_importance"],
                outputs=["feature_importance", "base_feature_importance"],
                name="feature_importance",
            ),
        ]
    )
