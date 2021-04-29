"""
A simple script to check that the connection between CI/CD runner (e.g. Drone)
and the shared storage volume on Minio is working correctly.
"""

from pathlib import Path
import os
from typing import Union

PATH_MINIO_DATA = os.getenv("MINIO_DATA_FOLDER")  # from Drone
PATH_TEST_FILE = Path(PATH_MINIO_DATA) / "raw" / "test.txt"


def read_text_file(filepath: Union[str, Path]) -> str:
    """
    Reads a text file and returns its contents as string.
    """
    with open(filepath, 'r') as file:
        text = file.read()
    
    return text


if __name__ == "__main__":

    text = read_text_file(PATH_TEST_FILE)
    print(text)
    