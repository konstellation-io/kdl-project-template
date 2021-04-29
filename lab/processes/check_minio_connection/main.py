from pathlib import Path
import os


PATH_MINIO_DATA = os.getenv("MINIO_DATA_FOLDER")  # from Drone
PATH_TEST_FILE = Path(PATH_MINIO_DATA) / "raw" / "test.txt"


def read_text_file(filepath):
    
    with open(filepath, 'r') as file:
        text = file.read()
    
    return text


if __name__ == "__main__":

    text = read_text_file(PATH_TEST_FILE)
    print(text)
    