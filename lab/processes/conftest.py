"""
Configuration for pytest unit tests
"""

import shutil

import pytest
from processes.prepare_data.cancer_data import prepare_cancer_data

from lib.utils import load_params


@pytest.fixture(name="config_test", scope="module")
def load_test_config():
    """Load configuration file for testing

    Yields:
        config: A ConfigBox object holding all the parameters needed
    """
    config = load_params("params.yaml")

    yield config.test


@pytest.fixture(name="temp_data_dir", scope="module")
def temporary_cancer_data_directory(config_test):
    """
    Pytest fixture for those tests that require a data directory containing the cancer dataset arrays.
    As part of setup, the fixture creates those arrays in the temporary location specified by the test parameters

    Keyword Arguments:
        dir_temp {str} -- Path where the files will be temporarily generated; the directory is cleared up after
            running the test (default: {"temp"})
    """

    dir_temp = config_test.paths.dir_temp
    dir_data_processed = config_test.paths.dir_processed
    dir_data_raw = config_test.paths.dir_raw

    # Setup:
    prepare_cancer_data(dir_input=dir_data_raw, dir_output=dir_data_processed)

    yield dir_data_processed

    # Teardown:
    shutil.rmtree(dir_temp)
