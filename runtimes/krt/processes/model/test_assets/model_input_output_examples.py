"""
An example of node input and output messages
(etl output before feeding to the model, and the resulting model output)
based on a single data point from the training set.
IMPROVE: Replace single-example equality check with distribution tests for the dataset before and after scaling
(otherwise fails upon retraining the model)
"""

from private_pb2 import EtlOutput, ModelOutput

etl_output_example = EtlOutput()
etl_output_example.scaled_features.mean_radius = -0.362939
etl_output_example.scaled_features.mean_texture = -0.089591
etl_output_example.scaled_features.mean_perimeter = -0.389856
etl_output_example.scaled_features.mean_area = -0.429621
etl_output_example.scaled_features.mean_smoothness = 0.033133
etl_output_example.scaled_features.mean_compactness = -0.443874
etl_output_example.scaled_features.mean_concavity = -0.749524
etl_output_example.scaled_features.mean_concave_points = -0.784241
etl_output_example.scaled_features.mean_symmetry = -0.172497
etl_output_example.scaled_features.mean_fractal_dimension = -0.132872
etl_output_example.scaled_features.radius_error = -0.665153
etl_output_example.scaled_features.texture_error = 0.085722
etl_output_example.scaled_features.perimeter_error = -0.652759
etl_output_example.scaled_features.area_error = -0.524728
etl_output_example.scaled_features.smoothness_error = -0.475154
etl_output_example.scaled_features.compactness_error = -0.492919
etl_output_example.scaled_features.concavity_error = -0.516343
etl_output_example.scaled_features.concave_points_error = -0.295416
etl_output_example.scaled_features.symmetry_error = -0.193994
etl_output_example.scaled_features.fractal_dimension_error = -0.582714
etl_output_example.scaled_features.worst_radius = -0.501247
etl_output_example.scaled_features.worst_texture = 0.069166
etl_output_example.scaled_features.worst_perimeter = -0.555430
etl_output_example.scaled_features.worst_area = -0.515767
etl_output_example.scaled_features.worst_smoothness = -0.096518
etl_output_example.scaled_features.worst_compactness = -0.422536
etl_output_example.scaled_features.worst_concavity = -0.681370
etl_output_example.scaled_features.worst_concave_points = -0.458166
etl_output_example.scaled_features.worst_symmetry = -0.229786
etl_output_example.scaled_features.worst_fractal_dimension = -0.599986

model_output_example = ModelOutput()
model_output_example.category = True
model_output_example.probability = 0.6
