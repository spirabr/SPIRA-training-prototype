from spira.adapter.config import load_config
from spira.adapter.random import initialize_random
from spira.adapter.valid_path import ValidPath, read_valid_paths_from_csv
from spira.core.domain.audio import load_audios
from spira.core.domain.dataset import SpiraDataset
from spira.core.domain.enum import OperationMode
from spira.core.domain.max_seq_length_calculator import AudiosLengthCalculator
from spira.core.domain.model import build_spira_model
from spira.core.domain.noise_generator import NoiseGenerator
from spira.core.domain.padding_inserter import PaddingGenerator
from spira.core.services.audio_processing import create_audio_processor

# dividir em duas partes inicializacao e runtime.
# iniciali\acao: try catch - Lendo os roles
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

# Feature engineering
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

audio_processor = create_audio_processor(config.audio_processor)
inputs = audio_processor.transform_into_features(inputs)

if config.feature_engineering.padding_with_max_length:
    audios_length_calculator = AudiosLengthCalculator(config.audio_processor)
    max_audio_length, _ = audios_length_calculator.calculate_min_max_audio_length(
        inputs
    )

    padding_generator = PaddingGenerator(max_audio_length)
    inputs = padding_generator.add_padding_to_features(inputs)

# TODO: Falta extrair o _get_feature_and_target_using_overlapping

# Training
################################################################################

# TODO: CLASS DATASET
dataset = SpiraDataset({"inputs": inputs, "labels": labels})

X_train, X_test, y_train, y_test = dataset.train_and_test_split_dataset()

model = build_spira_model(config)

# TODO: FIT
# trained_model = fit(model, X_train, y_train, X_test)

# TODO: Model Validation

# validate_model(trained_model)
