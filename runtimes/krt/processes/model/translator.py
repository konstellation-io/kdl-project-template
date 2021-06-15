"""
The translator offers a convenient way to convert 
between protobuff messages and numpy arrays or other 
object types
"""

import numpy as np

from private_pb2 import EtlOutput
from public_input_pb2 import InputFeatures


def etl_output_to_array(etl_output: InputFeatures) -> np.ndarray:
    """
    Translates a protocol buffer message of type EtlOutput to a numpy array

    Arguments:
        message {EtlOutput} -- Protocol buffer message (EtlOutput) to be converted

    Returns:
        [np.ndarray] -- Output numpy array
    """
    features_as_list = [
        etl_output.mean_radius,
        etl_output.mean_texture,
        etl_output.mean_perimeter,
        etl_output.mean_area,
        etl_output.mean_smoothness,
        etl_output.mean_compactness,
        etl_output.mean_concavity,
        etl_output.mean_concave_points,
        etl_output.mean_symmetry,
        etl_output.mean_fractal_dimension,
        etl_output.radius_error,
        etl_output.texture_error,
        etl_output.perimeter_error,
        etl_output.area_error,
        etl_output.smoothness_error,
        etl_output.compactness_error,
        etl_output.concavity_error,
        etl_output.concave_points_error,
        etl_output.symmetry_error,
        etl_output.fractal_dimension_error,
        etl_output.worst_radius,
        etl_output.worst_texture,
        etl_output.worst_perimeter,
        etl_output.worst_area,
        etl_output.worst_smoothness,
        etl_output.worst_compactness,
        etl_output.worst_concavity,
        etl_output.worst_concave_points,
        etl_output.worst_symmetry,
        etl_output.worst_fractal_dimension
    ]

    input_array = np.array(features_as_list)

    return input_array
