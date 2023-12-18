import json

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from spira.adapter.valid_path import ValidPath
from spira.core.services.audio_processing import AudioProcessorType


@dataclass
class DatasetConfig:
    normalize: bool
    # eval_csv: ValidPath
    # test_csv: ValidPath
    # inf_csv: ValidPath
    patients_csv: ValidPath
    controls_csv: ValidPath
    noises_csv: ValidPath


@dataclass
class ModelConfig:
    name: str
    fc1_dim: int
    fc2_dim: int


@dataclass
class DataAugmentationConfig:
    insert_noise: bool
    num_noise_control: int
    num_noise_patient: int
    noise_max_amp: float
    noise_min_amp: float


@dataclass
class FeatureEngineeringConfig:
    split_wav_using_overlapping: bool
    window_len: int
    step: int
    padding_with_max_length: bool
    hop_length: int


@dataclass
class TestConfig:
    batch_size: int
    num_workers: int


@dataclass
class TrainConfig:
    early_stop_epochs: int
    lr_decay: bool
    warmup_steps: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    loss1_weight: float
    batch_size: int
    seed: int
    num_workers: int
    logs_path: ValidPath
    reinit_layers: None
    summary_interval: int
    checkpoint_interval: int


@dataclass
class AudioProcessorConfig:
    feature_type: AudioProcessorType
    sample_rate: int
    num_mels: int
    mel_fmin: float
    mel_fmax: None
    num_mfcc: int
    log_mels: bool
    n_fft: int
    num_freq: int
    hop_length: int
    win_length: int


class Config(BaseModel):
    seed: int
    dataset: DatasetConfig
    feature_engineering: FeatureEngineeringConfig
    model: ModelConfig
    data_augmentation: DataAugmentationConfig
    test_config: TestConfig
    train_config: TrainConfig
    audio_processor: AudioProcessorConfig


def read_config(config_path: ValidPath) -> Config:
    with open(config_path, "r") as file:
        config_json = json.load(file)
    return Config(**config_json)


def validate_alternative_options_or_raise(config: Config):
    # xor operator - just one of these should be available
    if (
        config.feature_engineering.padding_with_max_length
        ^ config.feature_engineering.split_wav_using_overlapping
    ):
        RuntimeError(
            "You cannot use the padding_with_max_length option in conjunction with the split_wav_using_overlapping option, disable one of them !!"
        )


def validate_feature_type(config: Config):
    valid_feature_types = ["spectrogram", "melspectrogram", "mfcc"]
    feature_type = config.audio_processor.feature_type
    if feature_type not in valid_feature_types:
        raise ValueError(f"Invalid Feature type: {str(feature_type)}")


def load_config(config_path: ValidPath) -> Config:
    config = read_config(config_path)
    validate_alternative_options_or_raise(config)
    validate_feature_type(config)
    return config
