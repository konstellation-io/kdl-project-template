"""
An example of input features before and after scaling 
based on a single data point from the training set.
IMPROVE: Replace single-example equality check with distribution tests for the dataset before and after scaling
(otherwise fails upon updating train set)
"""

from public_input_pb2 import InputFeatures

input_features_example = InputFeatures()
input_features_example.mean_radius = 12.76
input_features_example.mean_texture = 18.84
input_features_example.mean_perimeter = 81.87
input_features_example.mean_area = 496.6
input_features_example.mean_smoothness = 0.09676
input_features_example.mean_compactness = 0.07952
input_features_example.mean_concavity = 0.02688
input_features_example.mean_concave_points = 0.01781
input_features_example.mean_symmetry = 0.1759
input_features_example.mean_fractal_dimension = 0.06183
input_features_example.radius_error = 0.2213
input_features_example.texture_error = 1.285
input_features_example.perimeter_error = 1.535
input_features_example.area_error = 17.26
input_features_example.smoothness_error = 0.005608
input_features_example.compactness_error = 0.01646
input_features_example.concavity_error = 0.01529
input_features_example.concave_points_error = 0.009997
input_features_example.symmetry_error = 0.01909
input_features_example.fractal_dimension_error = 0.002133
input_features_example.worst_radius = 13.75
input_features_example.worst_texture = 25.99
input_features_example.worst_perimeter = 87.82
input_features_example.worst_area = 579.7
input_features_example.worst_smoothness = 0.1298
input_features_example.worst_compactness = 0.1839
input_features_example.worst_concavity = 0.1255
input_features_example.worst_concave_points = 0.08312
input_features_example.worst_symmetry = 0.2744
input_features_example.worst_fractal_dimension = 0.07238

scaled_features_example = InputFeatures()
scaled_features_example.mean_radius = -0.36293972439160777
scaled_features_example.mean_texture = -0.08959166470374169
scaled_features_example.mean_perimeter = -0.3898568222406468
scaled_features_example.mean_area = -0.4296216699349167
scaled_features_example.mean_smoothness = 0.033133520375567826
scaled_features_example.mean_compactness = -0.4438743980871103
scaled_features_example.mean_concavity = -0.7495249160498719
scaled_features_example.mean_concave_points = -0.7842417525184496
scaled_features_example.mean_symmetry = -0.17249748324309333
scaled_features_example.mean_fractal_dimension = -0.13287216133939642
scaled_features_example.radius_error = -0.6651535167976715
scaled_features_example.texture_error = 0.08572291204755497
scaled_features_example.perimeter_error = -0.6527598918968666
scaled_features_example.area_error = -0.5247287573638606
scaled_features_example.smoothness_error = -0.4751540829274855
scaled_features_example.compactness_error = -0.4929196811534755
scaled_features_example.concavity_error = -0.5163437210453715
scaled_features_example.concave_points_error = -0.2954164956199253
scaled_features_example.symmetry_error = -0.19399427394083535
scaled_features_example.fractal_dimension_error = -0.5827142717526937
scaled_features_example.worst_radius = -0.5012477365846845
scaled_features_example.worst_texture = 0.06916688363473451
scaled_features_example.worst_perimeter = -0.5554303414894393
scaled_features_example.worst_area = -0.5157679413650915
scaled_features_example.worst_smoothness = -0.09651875340761147
scaled_features_example.worst_compactness = -0.4225367250862155
scaled_features_example.worst_concavity = -0.6813702577321857
scaled_features_example.worst_concave_points = -0.4581660790608254
scaled_features_example.worst_symmetry = -0.22978697376488608
scaled_features_example.worst_fractal_dimension = -0.59998672638929
