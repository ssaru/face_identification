import random
from pathlib import Path
from typing import Dict, List, Union


def write_meta(filepath: Path, filelist: List):
    with filepath.open("w", encoding="utf-8") as f:
        for filepath in filelist:
            imagepath = filepath
            _id = filepath.parent.name
            label = 0
            f.write(f"{imagepath},{0},{_id}\n")


if __name__ == "__main__":
    valid_ratio = 0.2

    dataset_path = Path("./data/lfw/Extracted_Faces")
    filelist = list(dataset_path.glob("**/*.jpg"))

    unique_ids = {filepath.parent.name: [filepath] for filepath in filelist}

    random.shuffle(filelist)
    num_train = int(len(filelist) * (1 - valid_ratio))

    train_filelist = filelist[:num_train]
    valid_filelist = filelist[num_train:]

    train_meta_filepath = Path("./train_data.txt")
    valid_meta_filepath = Path("./valid_data.txt")
    write_meta(filepath=train_meta_filepath, filelist=train_filelist)
    write_meta(filepath=valid_meta_filepath, filelist=valid_filelist)
