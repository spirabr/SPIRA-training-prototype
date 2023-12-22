import os
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore
from pydantic import BaseModel, model_serializer, model_validator


class ValidPath(BaseModel, PathLike[str]):
    path: Path

    def __fspath__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)

    @classmethod
    def from_str(cls, data: str):
        return ValidPath.model_construct(path=create_valid_path(data))

    @classmethod
    @model_validator(mode="before")
    def read(cls, data: Any) -> dict[str, Path]:
        if isinstance(data, str):
            return {"path": create_valid_path(data)}
        if isinstance(data, dict):
            return data
        raise RuntimeError("Path should be a string.")

    @model_serializer()
    def write(self) -> Path:
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
    df = pd.read_csv(str(csv_path), header=None, sep=",")
    # We are assuming the first column is the paths column
    paths_list = df[0].tolist()
    valid_paths_list = [ValidPath.from_str(path) for path in paths_list]
    return valid_paths_list
