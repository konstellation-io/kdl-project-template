"""
Tests for krt/etl/main.py
"""


import os
from unittest import mock

import joblib
import numpy as np
import pytest

from main import PATH_SCALER, handler, init
from private_pb2 import EtlOutput
from public_input_pb2 import InputFeatures
from test_assets.input_features_examples import (input_features_example,
                                                 scaled_features_example)
from translator import input_features_to_array


class CtxMock:
    """
    TODO: Docstring
    """
    logger = mock.Mock()
    set = mock.Mock()
    get = mock.Mock()

    def path(self, p):
        return os.path.join("runtimes/krt", p)


class DataMock:
    """TODO: Check if OK here or needs to be inside test_handler
    """
    def Unpack(self, inputs: InputFeatures):
        inputs.CopyFrom(input_features_example)


def test_init():
    """
    TODO: Docstring
    """
    ctx = CtxMock()
    init(ctx)

    ctx.set.assert_called_once_with("scaler", mock.ANY)


@pytest.mark.asyncio
async def test_handler():
    """
    TODO: Docstring
    """
    ctx = CtxMock()
    ctx.get.return_value = joblib.load(ctx.path(PATH_SCALER))

    # Define expected output object
    expected_result = EtlOutput()
    # pylint: disable=no-member
    DataMock().Unpack(expected_result.input_features)
    expected_result.input_features.CopyFrom(input_features_example)
    expected_result.scaled_features.CopyFrom(scaled_features_example)

    # Compute results with handler
    result = await handler(ctx, DataMock())
    result_array = input_features_to_array(result.scaled_features)
    expected_array = input_features_to_array(expected_result.scaled_features)

    # Check resulting outputs
    assert np.allclose(result_array, expected_array, rtol=1e-4)
    assert result.input_features == expected_result.input_features
