import json
from pathlib import Path

from spira.adapter.config import load_config
from spira.adapter.random_adapter import Random, TestRandom, TrainRandom
from spira.adapter.read_data import (
    read_noises_list_csv,
    read_patients_and_controls_list_csv,
)
from spira.core.domain.enum import OperationMode
from spira.core.domain.model import build_spira_model
from spira.core.domain.noise_generator import NoiseGenerator
from spira.core.services.audio_processing import AudioProcessor
from spira.core.services.data_augmentation_service import generate_noisy_audios
from spira.core.services.error_validation import (
    validate_config_is_valid,
    validate_files_exist,
)

###### Pre-config #####

config_path = Path("/app/spira/spira.json")

with open(config_path, "r") as file:
    config_json = json.load(file)

print(config_json)

config = load_config(config_json)

operation_mode = OperationMode.TRAIN

validate_config_is_valid(config)
validate_files_exist(config)

audio_processor = AudioProcessor(config.audio)


###### Data extraction ######

patients_list, controls_list = read_patients_and_controls_list_csv(
    config.dataset.test_csv
)
# todo: não sei como fazer essa função
noises_list = read_noises_list_csv(config.dataset.noise_csv)


###### Feature engineering ######

patients = audio_processor.load_audio_from_list(patients_list)
controls = audio_processor.load_audio_from_list(controls_list)
noises = audio_processor.load_audio_from_list(noises_list)


def initialize_random(config, operation_mode) -> Random:
    match operation_mode:
        case OperationMode.TRAIN:
            return TrainRandom(config.seed)
        case OperationMode.TEST:
            return TestRandom(config.seed)
        case _:
            raise RuntimeError("Bla")


randomizer = initialize_random(config, operation_mode)

# todo: validar com renato o uso de NoiseGenerator e não uso do Random no generate_noisy...
noise_generator = NoiseGenerator(
    noises,
    config.data_augmentation.noise_min_amp,
    config.data_augmentation.noise_max_amp,
    randomizer,
)

noisy_patients = generate_noisy_audios(
    patients, config.data_augmentation.num_noise_patient, noise_generator
)
noisy_controls = generate_noisy_audios(
    controls, config.data_augmentation.num_noise_control, noise_generator
)

# todo: put noisy audios into one dataset


###### train and test data split ######

###### Hyperparameters configuration #####
model = build_spira_model(config)

##### Train the model #####

##### Validate the model #####

##### Analytics #####
