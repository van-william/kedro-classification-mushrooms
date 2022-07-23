"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_mushrooms, normalize_data, mushrooms_raw, documentation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
#    node(
#                func=documentation,
#                inputs="data_context_raw",
#                outputs="data_context_md",
#                name="documentation_cleanup",
#    ),
#    node(
#                func=mushrooms_raw,
#                inputs="mushrooms",
#                outputs="raw_mushrooms",
#                name="dump_raw_mushrooms_node",
#   ),
    node(
                func=preprocess_mushrooms,
                inputs="raw_mushrooms",
                outputs="preprocessed_mushrooms",
                name="preprocess_mushrooms_node",
    ),
    node(
                func=normalize_data,
                inputs="preprocessed_mushrooms",
                outputs="normalized_mushrooms",
                name="normalize_mushrooms_node",
    )
        ]
    )

