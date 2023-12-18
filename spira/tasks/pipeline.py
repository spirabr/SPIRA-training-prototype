from spira.core.services.audio_processing import create_audio_processor

from spira.adapter.config import load_config
from spira.adapter.random import initialize_random
from spira.adapter.valid_path import ValidPath, read_valid_paths_from_csv
from spira.core.domain.audio import load_audios
from spira.core.domain.dataloader import (
    create_test_dataloader,
    create_train_dataloader,
)
from spira.core.domain.dataset import create_train_and_test_datasets
from spira.core.domain.enum import OperationMode
from spira.core.domain.model import build_spira_model
from spira.core.services.audio_feature_transformer import (
    create_audio_feature_transformer,
)
from spira.core.services.noise_generator import NoiseGenerator

# dividir em duas partes inicializacao e runtime.
# inicialização: try catch - Lendo os roles
# Ja as excecoes durante o processamento/runtime a gente pensa de outra maneira.

# Setup
################################################################################

config_path = ValidPath.from_str("/app/spira/spira.json")
config = load_config(config_path)

operation_mode = OperationMode.TRAIN
randomizer = initialize_random(config, operation_mode)

# Data Loading
################################################################################

patients_paths = read_valid_paths_from_csv(config.dataset.patients_csv)
controls_paths = read_valid_paths_from_csv(config.dataset.controls_csv)
noises_paths = read_valid_paths_from_csv(config.dataset.noises_csv)

patients = load_audios(patients_paths, config.dataset.normalize)
controls = load_audios(controls_paths, config.dataset.normalize)
noises = load_audios(noises_paths, config.dataset.normalize)

# Data Processing
################################################################################

# We are assuming all the patients have the disease.
label_patients = [1 for _ in range(len(patients))]
label_controls = [0 for _ in range(len(controls))]
labels = label_patients + label_controls

if config.data_augmentation.insert_noise:
    noise_generator = NoiseGenerator(
        noises,
        config.data_augmentation.noise_min_amp,
        config.data_augmentation.noise_max_amp,
        randomizer,
    )

    patients = noise_generator.generate_noisy_audios(
        config.data_augmentation.num_noise_patient, patients
    )
    controls = noise_generator.generate_noisy_audios(
        config.data_augmentation.num_noise_control, controls
    )

inputs = patients + controls

# Feature Engineering
################################################################################

audio_processor = create_audio_processor(config.audio_processor)
audio_feature_transformer = create_audio_feature_transformer(
    audio_processor, config.feature_engineering
)
features = audio_feature_transformer.transform_into_features(inputs)

# Dataset Generation
################################################################################

train_dataset, test_dataset = create_train_and_test_datasets(
    features, labels, config.seed
)

train_data_loader = create_train_dataloader(
    train_dataset, config.train_config.batch_size, config.train_config.num_workers
)
test_data_loader = create_test_dataloader(
    test_dataset, config.test_config.batch_size, config.test_config.num_workers
)

# Training
################################################################################

model = build_spira_model(config)

# TODO: FIT
# trained_model = fit(model, X_train, y_train, X_test)

# TODO: Model Validation

# validate_model(trained_model)
