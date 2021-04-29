from pathlib import Path
import os


PATH_MINIO_DATA = os.getenv("MINIO_DATA_FOLDER")  # from Drone
PATH_TEST_FILE = Path(PATH_MINIO_DATA) / "raw" / "test.txt"


if __name__ == "__main__":

    with open(PATH_TEST_FILE, 'r') as file:
        text = file.read()
    
    print(text)