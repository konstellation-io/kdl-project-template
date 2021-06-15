"""
Tests for krt/model/main.py
"""

import os
from unittest import mock

import joblib
import numpy as np
import pytest

from main import PATH_MODEL, handler, init
from private_pb2 import EtlOutput, ModelOutput
from test_assets.model_input_output_examples import (etl_output_example,
                                                     model_output_example)


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
    def Unpack(self, inputs: EtlOutput):
        inputs.CopyFrom(etl_output_example)


def test_init():
    """
    TODO: Docstring
    """
    ctx = CtxMock()
    init(ctx)

    ctx.set.assert_called_once_with("model", mock.ANY)


@pytest.mark.asyncio
async def test_handler():
    
    ctx = CtxMock()
    ctx.get.return_value = joblib.load(ctx.path(PATH_MODEL))

    # Define expected output object
    expected_result = model_output_example
    # pylint: disable=no-member

    # Compute the model output with function under test
    result = await handler(ctx=ctx, data=DataMock())

    # Check resulting outputs against expected
    assert result.category == expected_result.category
    assert result.probability > 0
    assert result.probability < 1
    assert np.isclose(result.probability, expected_result.probability, rtol=1e-3)
