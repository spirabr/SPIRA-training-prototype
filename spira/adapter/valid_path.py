import os
from os import PathLike
from pathlib import Path

import pandas as pd


class ValidPath(PathLike[str]):
    def __init__(self, unvalidated_path: str):
        self.path = create_valid_path(unvalidated_path)

    def __fspath__(self):
        return self.path


def check_file_exists(path: Path) -> bool:
    return os.path.isfile(path)


def check_file_exists_or_raise(path: Path):
    if not check_file_exists(path):
        raise FileExistsError(f"File {path} does not exist.")


def create_valid_path(unvalidated_path: str) -> Path:
    path = Path(unvalidated_path)
    check_file_exists_or_raise(path)
    return path


def read_valid_paths_from_csv(csv_path: ValidPath) -> list[ValidPath]:
    return pd.read_csv(csv_path, sep=",").map(ValidPath).to_list
