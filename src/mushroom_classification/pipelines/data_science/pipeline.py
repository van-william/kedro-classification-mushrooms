"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, iterative_training, simple_decision_tree


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["normalized_mushrooms", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="classifier_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
            node(
                func=iterative_training,
                inputs=["preprocessed_mushrooms", "base_feature_importance", "params:model_options"],
                outputs= "summary_results",
                name="iterative_training"
            ),
            node(
                func=simple_decision_tree,
                inputs=["preprocessed_mushrooms", "base_feature_importance", "params:model_options"],
                outputs= "simple_classifier_model",
                name="tree_training"
            ),
        ]
    )