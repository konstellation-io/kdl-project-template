
import joblib
from sklearn.base import BaseEstimator

from private_pb2 import EtlOutput, ModelOutput
from test_assets.model_input_output_examples import (etl_output_example,
                                                     model_output_example)
from translator import etl_output_to_array

PATH_MODEL = "assets/model.joblib"


def init(ctx):
    """
    TODO: Docstring

    Arguments:
        ctx {[type]} -- [description]
    """
    ctx.logger.info('Initializing runner')

    model_path = ctx.path(PATH_MODEL)
    ctx.logger.info(f'Loading scaler from {model_path}')

    model = joblib.load(model_path)
    ctx.set('model', model)
    ctx.logger.info('Model loaded')


async def handler(ctx, data) -> ModelOutput:
    """
    TODO: Docstring

    Arguments:
        ctx {[type]} -- [description]
        data {[type]} -- [description]

    Returns:
        EtlOutput -- [description]
    """
    req = EtlOutput()
    data.Unpack(req)
    model: BaseEstimator = ctx.get('model')
    return new_model_output(model, req)


# pylint: disable=no-member
def new_model_output(model: BaseEstimator, etl_output: EtlOutput) -> ModelOutput:
    result = ModelOutput()

    # Convert input to numpy array
    input_array = etl_output_to_array(etl_output.scaled_features).reshape(1, -1)

    # Make prediction (e.g. output_array = model.predict()
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)[0, -1]

    # Convert array back to protobuffer
    result.category = prediction
    result.probability = probability

    return result
