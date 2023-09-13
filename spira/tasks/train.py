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


def check_config_is_valid(config: SpiraConfig) -> [bool, str]:
    if config.padding_with_max_length ^ config.split_wav_using_overlapping:
        return [False,
                "You cannot use the padding_with_max_length option in conjunction with the split_wav_using_overlapping option, disable one of them !!"]


def check_files_exist(config: SpiraConfig) -> [bool, str]:
    if os.path.isfile(config.noise_csv):
        return [False, "Noise CSV file don't exists! Fix it in config.json"]
    if os.path.isfile(config.dataset_csv):
        return [False, "Test or Train CSV file don't exists! Fix it in config.json"]


def read_list_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=',')


def read_wav_from_list(file_list: pd.DataFrame) -> list[pd.DataFrame]:
    return file_list.map(lambda file: audio_processor.load_wav(file))


def load_spira_datasets_and_classes(dataset_csv_path: Path) -> list[[pd.DataFrame, str]]:
    dataset_list = read_list_csv(dataset_csv_path)
    datasets = read_wav_from_list(dataset_list[0])
    classes = dataset_list[1]
    return zip(datasets, classes)


def load_spira_datasets(dataset_csv_path: Path) -> list[pd.DataFrame]:
    dataset_list = read_list_csv(dataset_csv_path)
    return read_wav_from_list(dataset_list[0])


def load_spira_noise(noise_csv_path: Path) -> list[pd.DataFrame]:
    noise_list = read_list_csv(noise_csv_path)
    return read_wav_from_list(noise_list)


is_config_valid = check_config_is_valid(config)
if not is_config_valid[0]:
    raise RuntimeError(is_config_valid[1])

do_files_exist = check_files_exist(config)
if not do_files_exist[0]:
    raise FileExistsError(do_files_exist[1])

check_files_exist(config)
tuples_dataset_and_class = load_spira_datasets_and_classes(config.dataset_csv)
noise = load_spira_noise(config.noise_csv)

###### Data extraction ######

###### train and test data split ######

###### Feature engineering ######


###### Hyperparameters configuration #####
model = build_spira_model(config)

##### Train the model #####

##### Validate the model #####

##### Analytics #####
