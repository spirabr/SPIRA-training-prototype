from spira.adapter.config import load_config
from spira.adapter.random import initialize_random
from spira.adapter.valid_path import ValidPath, read_valid_paths_from_csv
from spira.core.domain.audio import Audios
from spira.core.domain.audio_processor import create_audio_processor
from spira.core.domain.checkpoint import create_checkpoint_builder
from spira.core.domain.cnn_builder import create_cnn_builder
from spira.core.domain.dataloader import (
    create_test_dataloader,
    create_train_dataloader,
)
from spira.core.domain.dataset import create_train_and_test_datasets
from spira.core.domain.enum import OperationMode
from spira.core.domain.loss import (
    create_test_loss_calculator,
    create_train_loss_calculator,
)
from spira.core.domain.model import create_model
from spira.core.domain.optimizer import create_optimizer
from spira.core.domain.scheduler import create_scheduler
from spira.core.services.audio_feature_transformer import (
    create_audio_feature_transformer,
)
from spira.core.services.model_trainer import BasicModelTrainer

# Setup
################################################################################

config_path = ValidPath.from_str("/app/spira/spira.json")
config = load_config(config_path)

operation_mode = OperationMode.TRAIN
randomizer = initialize_random(config, operation_mode)

# Data Loading
################################################################################

patients_paths = read_valid_paths_from_csv(config.parameters.dataset.patients_csv)
controls_paths = read_valid_paths_from_csv(config.parameters.dataset.controls_csv)
noises_paths = read_valid_paths_from_csv(config.parameters.dataset.noises_csv)

patients_inputs = Audios.load(
    patients_paths,
    config.parameters.audio.hop_length,
    config.parameters.dataset.normalize,
)
controls_inputs = Audios.load(
    controls_paths,
    config.parameters.audio.hop_length,
    config.parameters.dataset.normalize,
)
noises = Audios.load(
    noises_paths,
    config.parameters.audio.hop_length,
    config.parameters.dataset.normalize,
)

# Feature Engineering
################################################################################

audio_processor = create_audio_processor(config.parameters.audio)

patient_feature_transformer = create_audio_feature_transformer(
    randomizer,
    audio_processor,
    config.options.feature_engineering,
    config.parameters.feature_engineering,
    config.parameters.feature_engineering.noisy_audio.num_noise_control,
    noises,
)

control_feature_transformer = create_audio_feature_transformer(
    randomizer,
    audio_processor,
    config.options.feature_engineering,
    config.parameters.feature_engineering,
    config.parameters.feature_engineering.noisy_audio.num_noise_control,
    noises,
)

patients_features = patient_feature_transformer.transform_into_features(patients_inputs)
controls_features = control_feature_transformer.transform_into_features(controls_inputs)

patients_label = [1 for _ in range(len(patients_features))]
controls_label = [0 for _ in range(len(controls_features))]

# Dataset Generation
################################################################################

features = patients_features + controls_features
labels = patients_label + controls_label

train_dataset, test_dataset = create_train_and_test_datasets(
    features, labels, config.seed
)

train_data_loader = create_train_dataloader(
    train_dataset,
    config.parameters.training.batch_size,
    config.parameters.training.num_workers,
)
test_data_loader = create_test_dataloader(
    test_dataset,
    config.parameters.training.batch_size,
    config.parameters.training.num_workers,
)

# Training
################################################################################

cnn_builder = create_cnn_builder(
    config.options.feature_engineering,
    config.parameters,
    train_dataset.features,
)

model = create_model(config.options.training, cnn_builder)

optimizer = create_optimizer(
    config.parameters.training.optimizer,
    model.get_parameters(),
)

scheduler = create_scheduler(config.parameters.training.scheduler, optimizer)

train_loss_calculator = create_train_loss_calculator(config.options.training)

test_loss_calculator = create_test_loss_calculator(config.options.training)

checkpoint_builder = create_checkpoint_builder(config.parameters.training.checkpoint)

model_trainer = BasicModelTrainer(
    model,
    optimizer,
    scheduler,
    train_loss_calculator,
    test_loss_calculator,
    checkpoint_builder,
)

previous_checkpoint = None
trained_model = model_trainer.train(
    train_data_loader,
    test_data_loader,
    config.parameters.training.epochs,
    previous_checkpoint,
)
