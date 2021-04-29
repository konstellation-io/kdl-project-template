from pathlib import Path
import os


# DIR_REPO = Path.cwd().parent.parent.parent
# DIR_DATA = DIR_REPO.parent / "shared-storage" / "kdl-project-template" / "data"

PATH_MINIO_DATA = os.getenv("MINIO_DATA_FOLDER")  # from Drone
PATH_TEST_FILE = Path(DIR_DATA) / "raw" / "test.txt"


if __name__ == "__main__":

    with open(PATH_TEST_FILE, 'r') as file:
        text = file.read()
    
    print(text)