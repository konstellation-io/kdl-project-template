"""
Reusable functions using the MLflow API
"""

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def get_best_run(
    mlflow_uri: str, 
    exp_name: str, 
    filter_string: str, 
    metric: str, 
    highest: bool
        ) -> mlflow.entities.Run:
    """
    Given an MLflow tracking URI and experiment name, returns the best 
    run according to the metric specified, after filtering on runs as 
    specified by the query in filter_string.

    Args:
        mlflow_uri: (str) MLflow tracking URI
        exp_name: (str) MLflow experiment name
        filter_string: (str) argument to MlflowClient.search_runs method 
            filter string (e.g. 'metrics.accuracy > 0.6'). See documentation 
            at https://www.mlflow.org/docs/latest/search-syntax.html for 
            syntax information.
        metric: (str) name of the metric to use in determining best run
        highest: (bool) whether the metric should be as high as possible 
            (e.g. accuracy); for metrics that should be as low as possible 
            (e.g. error) use False
        
    Returns:
        (mlflow.entities.Run): implements properties:
            .data (including dicts .metrics and .params), and
            .info (including .run_id)

    Example:
        MLFLOW_URI = "http://mlflow-server:5000" 
        EXP_NAME = "kdl-project-template"
        METRIC_NAME = "val_acc"
        FILTER_STRING = "metrics.val_acc > 0.9"

        run = get_best_run(
            mlflow_uri=MLFLOW_URI, 
            exp_name=EXP_NAME, 
            filter_string=FILTER_STRING, 
            metric=METRIC_NAME, 
            highest=True
        )

    """
    client = MlflowClient(tracking_uri=mlflow_uri)
    exp_id = client.get_experiment_by_name(exp_name).experiment_id

    runs = client.search_runs(exp_id, filter_string=filter_string)
    runs_and_acc = [(r.data.metrics[metric], r.info.run_id) for r in runs]
    _, top_run_id = sorted(runs_and_acc, reverse=highest)[0]
    
    best_run = client.get_run(run_id=top_run_id)

    return best_run
