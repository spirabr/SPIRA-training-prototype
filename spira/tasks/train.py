from spira.adapter.config import load_config
from spira.adapter.random import initialize_random
from spira.adapter.valid_path import ValidPath, read_valid_paths_from_csv
from spira.core.domain.enum import OperationMode
from spira.core.domain.model import build_spira_model
from spira.core.domain.noise_generator import NoiseGenerator
from spira.core.services.audio_processing import AudioProcessor
from spira.core.services.data_augmentation import generate_noisy_audios

# dividir em duas partes inicializacao e runtime.
# iniciali\acao: try catch - Lendo os roles
# Ja as excecoes durante o processamento/runtime a gente pensa de outra maneira.

###### Pre-config #####

config_path = ValidPath.from_str("/app/spira/spira.json")
config = load_config(config_path)

operation_mode = OperationMode.TRAIN

audio_processor = AudioProcessor(config.audio)

###### Data extraction ######

patients_paths = read_valid_paths_from_csv(config.dataset.patients_csv)
controls_paths = read_valid_paths_from_csv(config.dataset.controls_csv)
noises_paths = read_valid_paths_from_csv(config.dataset.noises_csv)


###### Feature engineering ######

patients = audio_processor.load_audios(patients_paths)
controls = audio_processor.load_audios(controls_paths)
noises = audio_processor.load_audios(noises_paths)

randomizer = initialize_random(config, operation_mode)

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
print(model.conv)

# TODO: Como o edresson fez o fit? Ver no código dele

##### Train the model #####

##### Validate the model #####

##### Analytics #####
