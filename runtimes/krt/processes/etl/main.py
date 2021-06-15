"""
# TODO: Docstring
"""


import joblib
from sklearn.preprocessing import StandardScaler

from private_pb2 import EtlOutput
from public_input_pb2 import InputFeatures
from translator import array_to_input_features, input_features_to_array

PATH_SCALER = 'assets/scaler.joblib'


def init(ctx):
    """
    TODO: Docstring

    Arguments:
        ctx {[type]} -- [description]
    """
    ctx.logger.info('Initializing runner')

    scaler_path = ctx.path(PATH_SCALER)
    ctx.logger.info(f'Loading scaler from {scaler_path}')

    scaler: StandardScaler = joblib.load(scaler_path)
    ctx.set('scaler', scaler)
    ctx.logger.info('Scaler loaded')


async def handler(ctx, data) -> EtlOutput:
    """
    TODO: Docstring

    Arguments:
        ctx {[type]} -- [description]
        data {[type]} -- [description]

    Returns:
        EtlOutput -- [description]
    """
    req = InputFeatures()
    ctx.logger.info(f"Data type: {type(data)}")
    data.Unpack(req)
    scaler: StandardScaler = ctx.get('scaler')
    return new_etl_output(scaler, req)


# pylint: disable=no-member
def new_etl_output(scaler: StandardScaler, input_features: InputFeatures) -> EtlOutput:
    """
    TODO: Docstring

    Arguments:
        scaler {StandardScaler} -- [description]
        input_features {InputFeatures} -- [description]

    Returns:
        EtlOutput -- [description]
    """
    result = EtlOutput()
    result.input_features.CopyFrom(input_features)

    # Convert to numpy array
    input_array = input_features_to_array(input_features).reshape(1, -1)
    output_array = scaler.transform(input_array)[0]

    # Convert array back to protobuffer
    result.scaled_features.CopyFrom(array_to_input_features(output_array))

    return result
