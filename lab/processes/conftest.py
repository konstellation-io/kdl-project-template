"""
Configuration for pytest unit tests
"""

import shutil

from pathlib import Path
import pytest
import yaml
from yaml.loader import SafeLoader

from processes.prepare_data.cancer_data import prepare_cancer_data


@pytest.fixture(name="temp_data_dir", scope="module")
def temporary_cancer_data_directory():
    """
    Pytest fixture for those tests that require a data directory containing the cancer dataset arrays.
    As part of setup, the fixture creates those arrays in the temporary location specified by dir_temp

    Keyword Arguments:
        dir_temp {str} -- Path where the files will be temporarily generated; the directory is cleared up after
            running the test (default: {"temp"})
    """

    with open("params.yaml", "rb") as config_file:
        config = yaml.load(config_file, Loader=SafeLoader)
    dir_temp = config["test"]["paths"]["dir_temp"]
    Path(dir_temp).mkdir()
    dir_data_processed = config["test"]["paths"]["dir_processed"]

    # Setup:
    prepare_cancer_data(dir_output=dir_data_processed)

    yield dir_data_processed

    # Teardown:
    shutil.rmtree(dir_temp)
