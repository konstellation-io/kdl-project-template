"""
The translator offers a convenient way to convert 
between protobuff messages and numpy arrays or other 
object types
"""

import numpy as np

from public_input_pb2 import InputFeatures


def input_features_to_array(input_features: InputFeatures) -> np.ndarray:
    """
    Translates a protocol buffer message of type InputFeatures to a numpy array

    Arguments:
        message {InputFeatures} -- Protocol buffer message (InputFeatures) to be converted

    Returns:
        [np.ndarray] -- Output numpy array
    """
    features_as_list = [
        input_features.mean_radius,
        input_features.mean_texture,
        input_features.mean_perimeter,
        input_features.mean_area,
        input_features.mean_smoothness,
        input_features.mean_compactness,
        input_features.mean_concavity,
        input_features.mean_concave_points,
        input_features.mean_symmetry,
        input_features.mean_fractal_dimension,
        input_features.radius_error,
        input_features.texture_error,
        input_features.perimeter_error,
        input_features.area_error,
        input_features.smoothness_error,
        input_features.compactness_error,
        input_features.concavity_error,
        input_features.concave_points_error,
        input_features.symmetry_error,
        input_features.fractal_dimension_error,
        input_features.worst_radius,
        input_features.worst_texture,
        input_features.worst_perimeter,
        input_features.worst_area,
        input_features.worst_smoothness,
        input_features.worst_compactness,
        input_features.worst_concavity,
        input_features.worst_concave_points,
        input_features.worst_symmetry,
        input_features.worst_fractal_dimension
    ]

    input_array = np.array(features_as_list)

    return input_array


def array_to_input_features(array: np.ndarray) -> InputFeatures:
    """Converts numpy array to protocol buffer of type InputFeatures

    Arguments:
        array {np.ndarray} -- input

    Returns:
        [InputFeatures] -- output protocol buffer message
    """

    input_features = InputFeatures()

    input_features.mean_radius = array[0]
    input_features.mean_texture = array[1]
    input_features.mean_perimeter = array[2]
    input_features.mean_area = array[3]
    input_features.mean_smoothness = array[4]
    input_features.mean_compactness = array[5]
    input_features.mean_concavity = array[6]
    input_features.mean_concave_points = array[7]
    input_features.mean_symmetry = array[8]
    input_features.mean_fractal_dimension = array[9]
    input_features.radius_error = array[10]
    input_features.texture_error = array[11]
    input_features.perimeter_error = array[12]
    input_features.area_error = array[13]
    input_features.smoothness_error = array[14]
    input_features.compactness_error = array[15]
    input_features.concavity_error = array[16]
    input_features.concave_points_error = array[17]
    input_features.symmetry_error = array[18]
    input_features.fractal_dimension_error = array[19]
    input_features.worst_radius = array[20]
    input_features.worst_texture = array[21]
    input_features.worst_perimeter = array[22]
    input_features.worst_area = array[23]
    input_features.worst_smoothness = array[24]
    input_features.worst_compactness = array[25]
    input_features.worst_concavity = array[26]
    input_features.worst_concave_points = array[27]
    input_features.worst_symmetry = array[28]
    input_features.worst_fractal_dimension = array[29]

    return input_features
