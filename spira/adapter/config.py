import json

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from spira.adapter.valid_path import ValidPath


@dataclass
class ConfigDataset:
    split_wav_using_overlapping: bool
    window_len: int
    step: int
    padding_with_max_length: bool
    max_seq_len: int
    # eval_csv: ValidPath
    # test_csv: ValidPath
    # inf_csv: ValidPath
    patients_csv: ValidPath
    controls_csv: ValidPath
    noises_csv: ValidPath


@dataclass
class ConfigModel:
    name: str
    fc1_dim: int
    fc2_dim: int


@dataclass
class ConfigDataAugmentation:
    insert_noise: bool
    num_noise_control: int
    num_noise_patient: int
    noise_max_amp: float
    noise_min_amp: float


@dataclass
class ConfigTestConfig:
    batch_size: int
    num_workers: int


@dataclass
class ConfigAudio:
    feature: str
    sample_rate: int
    normalize: bool
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
    dataset: ConfigDataset
    model: ConfigModel
    data_augmentation: ConfigDataAugmentation
    test_config: ConfigTestConfig
    audio: ConfigAudio

    def num_features(self):
        if self.audio.feature == "spectrogram":
            return self.audio.num_freq
        elif self.audio.feature == "melspectrogram":
            return self.audio.num_mels
        elif self.audio.feature == "mfcc":
            return self.audio.num_mfcc
        return None


def read_config(config_path: ValidPath) -> Config:
    with open(config_path, "r") as file:
        config_json = json.load(file)
    return Config(**config_json)


def validate_alternative_options_or_raise(config: Config):
    # xor operator - just one of these should be available
    if (
        config.dataset.padding_with_max_length
        ^ config.dataset.split_wav_using_overlapping
    ):
        RuntimeError(
            "You cannot use the padding_with_max_length option in conjunction with the split_wav_using_overlapping option, disable one of them !!"
        )


def load_config(config_path: ValidPath) -> Config:
    config = read_config(config_path)
    validate_alternative_options_or_raise(config)
    return config
