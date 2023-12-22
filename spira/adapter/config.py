import json

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from spira.adapter.valid_path import ValidPath
from spira.core.domain.audio_processor import AudioProcessorType
from spira.core.domain.optimizer import OptimizerCategory


@dataclass
class FeatureEngineeringOptionsConfig:
    use_noise: bool
    use_overlapping: bool
    use_padding: bool
    use_mixture: bool


@dataclass
class TrainingOptionsConfig:
    use_lr_decay: bool
    use_clipping: bool
    use_class_balancing: bool


@dataclass
class OptionsConfig:
    feature_engineering: FeatureEngineeringOptionsConfig
    training: TrainingOptionsConfig


@dataclass
class DatasetParametersConfig:
    normalize: bool
    # eval_csv: ValidPath
    # test_csv: ValidPath
    # inf_csv: ValidPath
    patients_csv: ValidPath
    controls_csv: ValidPath
    noises_csv: ValidPath


@dataclass
class ModelParametersConfig:
    name: str
    fc1_dim: int
    fc2_dim: int


@dataclass
class NoisyAudioFeatureTransformerConfig:
    num_noise_control: int
    num_noise_patient: int
    noise_max_amp: float
    noise_min_amp: float


@dataclass
class OverlappedAudioFeatureTransformerConfig:
    window_length: int
    step_size: int


@dataclass
class MixedAudioFeatureTransformerConfig:
    alpha: float
    beta: float


@dataclass
class FeatureEngineeringParametersConfig:
    noisy_audio: NoisyAudioFeatureTransformerConfig
    overlapped_audio: OverlappedAudioFeatureTransformerConfig
    mixed_audio: MixedAudioFeatureTransformerConfig


@dataclass
class TestParametersConfig:
    batch_size: int
    num_workers: int


@dataclass
class TrainOptimizerConfig:
    category: OptimizerCategory
    learning_rate: float
    weight_decay: float


@dataclass
class TrainSchedulerConfig:
    use_lr_decay: bool
    warmup_steps: int


@dataclass
class TrainLossCalculatorConfig:
    use_class_balancing: bool


@dataclass
class TrainCheckpointConfig:
    dir: ValidPath
    interval: int


@dataclass
class TrainingParametersConfig:
    optimizer: TrainOptimizerConfig
    scheduler: TrainSchedulerConfig
    loss_calculator: TrainLossCalculatorConfig
    checkpoint: TrainCheckpointConfig
    early_stop_epochs: int
    epochs: int
    loss1_weight: float
    batch_size: int
    seed: int
    num_workers: int
    logs_path: ValidPath
    reinit_layers: None
    summary_interval: int


@dataclass
class MFCCAudioProcessorConfig:
    sample_rate: int
    num_mels: int
    num_mfcc: int
    log_mels: bool
    n_fft: int
    win_length: int


@dataclass
class SpectrogramAudioProcessorConfig:
    sample_rate: int
    num_mels: int
    mel_fmin: float
    mel_fmax: None
    num_mfcc: int
    log_mels: bool
    n_fft: int
    num_freq: int
    win_length: int


@dataclass
class MelspectrogramAudioProcessorConfig:
    sample_rate: int
    num_mels: int
    mel_fmin: float
    mel_fmax: None
    num_mfcc: int
    log_mels: bool
    n_fft: int
    num_freq: int
    win_length: int


@dataclass
class AudioProcessorParametersConfig:
    feature_type: AudioProcessorType
    hop_length: int
    mfcc: MFCCAudioProcessorConfig
    spectrogram: SpectrogramAudioProcessorConfig
    melspectrogram: MelspectrogramAudioProcessorConfig

    def num_features(self):
        match self.feature_type:
            case AudioProcessorType.MFCC:
                return self.mfcc.num_mfcc
            case AudioProcessorType.SPECTROGRAM:
                return self.spectrogram.num_freq
            case AudioProcessorType.MELSPECTROGRAM:
                return self.melspectrogram.num_mels


@dataclass
class ParametersConfig:
    audio: AudioProcessorParametersConfig
    dataset: DatasetParametersConfig
    feature_engineering: FeatureEngineeringParametersConfig
    model: ModelParametersConfig
    training: TrainingParametersConfig


class Config(BaseModel):
    seed: int
    options: OptionsConfig
    parameters: ParametersConfig


def read_config(config_path: ValidPath) -> Config:
    with open(config_path, "r") as file:
        config_json = json.load(file)
    return Config(**config_json)


def validate_consistent_options(options: OptionsConfig):
    if options.feature_engineering.use_mixture == options.training.use_clipping:
        RuntimeError("You should use a clipped loss when using mixed data")


def validate_feature_engineering_options_or_raise(
    options: FeatureEngineeringOptionsConfig,
):
    # xor operator - just one of these should be available
    if options.use_padding ^ options.use_overlapping:
        RuntimeError(
            "You cannot use paddings and overlapping options for feature engineering"
        )


def validate_feature_type(parameters: ParametersConfig):
    valid_feature_types = ["spectrogram", "melspectrogram", "mfcc"]
    feature_type = parameters.audio.feature_type
    if feature_type not in valid_feature_types:
        raise ValueError(f"Invalid Feature type: {str(feature_type)}")


def load_config(config_path: ValidPath) -> Config:
    config = read_config(config_path)
    validate_feature_engineering_options_or_raise(config.options.feature_engineering)
    validate_feature_type(config.parameters)
    return config
