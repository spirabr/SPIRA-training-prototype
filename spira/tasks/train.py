from pathlib import Path
import os

import pandas as pd

from spira.training.config import load_spira_config, SpiraConfig
from spira.training.model import build_spira_model
from spira.dataset.audio_processing import AudioProcessor


###### Pre-config #####
config = load_spira_config(Path("../spira.json"))

# TODO: configurar o audio processor
audio_processor = AudioProcessor()

### Validate files existance

# todo: Fazer as próprias exceções


def validate_config_is_valid(config: SpiraConfig):
    if config.padding_with_max_length ^ config.split_wav_using_overlapping:
        RuntimeError("You cannot use the padding_with_max_length option in conjunction with the split_wav_using_overlapping option, disable one of them !!")


def validate_files_exist(config: SpiraConfig):
    if os.path.isfile(config.noise_csv):
        FileExistsError("Noise CSV file don't exists! Fix it in config.json")
    if os.path.isfile(config.dataset_csv):
        FileExistsError("Test or Train CSV file don't exists! Fix it in config.json")


def validate_datasets_and_classes_are_compatible(file_list: pd.Series, classes: pd.Series):
    if len(file_list) != len(classes):
        RuntimeError("Dataset and classes have different lengths")

def read_list_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=',')


def read_wav_from_list(file_list: pd.DataFrame) -> list[pd.DataFrame]:
    return file_list.map(lambda file: audio_processor.load_wav(file))


def load_dataset_list(dataset_csv_path: Path) -> [pd.Series, pd.Series]:
    dataset_list = read_list_csv(dataset_csv_path)
    return dataset_list[0], dataset_list[1]


def load_spira_datasets(dataset_csv_path: Path) -> list[pd.DataFrame]:
    dataset_list = read_list_csv(dataset_csv_path)
    return read_wav_from_list(dataset_list[0])


def load_spira_noise(noise_csv_path: Path) -> list[pd.DataFrame]:
    noise_list = read_list_csv(noise_csv_path)
    return read_wav_from_list(noise_list)


validate_config_is_valid(config)
validate_files_exist(config)
file_list, classes = load_dataset_list(config.dataset_csv)
validate_datasets_and_classes_are_compatible(file_list, classes)
datasets = read_wav_from_list(file_list)
noise = load_spira_noise(config.noise_csv)

###### Data extraction ######

###### train and test data split ######

###### Feature engineering ######


###### Hyperparameters configuration #####
model = build_spira_model(config)

##### Train the model #####

##### Validate the model #####

##### Analytics #####
