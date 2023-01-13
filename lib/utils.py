"""
Miscellaneous utility functions
"""
import yaml
from box import ConfigBox


def flatten_list(input_list: list) -> list:
    """
    Flattens a nested list that contains lists as its elements.
    Only goes one level deep (i.e. works on lists of lists but not lists of lists of lists).
    """
    return [item for sublist in input_list for item in sublist]


def load_params(params_path: str) -> ConfigBox:
    """Load parameters file to be used as object

    Args:
        params_path (str): route to parameters

    Returns:
        _type_: _description_
    """
    with open(params_path, "rb") as config_file:
        params = yaml.safe_load(config_file)
        params = ConfigBox(params)
    return params
