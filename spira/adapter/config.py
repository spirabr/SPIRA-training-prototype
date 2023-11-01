from pathlib import Path

from pydantic import BaseModel
from pydantic.dataclasses import dataclass


@dataclass
class ConfigDataset:
    split_wav_using_overlapping: bool
    window_len: int
    step: int
    padding_with_max_length: bool
    max_seq_len: int
    eval_csv: Path
    eval_data_root_path: Path
    test_csv: Path
    test_data_root_path: Path
    inf_csv: Path
    inf_data_root_path: Path
    noise_csv: Path
    noise_data_root_path: Path
    control_class: int
    patient_class: int


@dataclass
class ConfigModel:
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
    model_name: str
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


def load_config(config_json) -> Config:
    return Config(
        model_name=config_json["model_name"],
        seed=config_json["seed"],
        dataset=config_json["dataset"],
        model=config_json["model"],
        data_augmentation=config_json["data_augmentation"],
        test_config=config_json["test_config"],
        audio=config_json["audio"],
    )


# def load_config(config_path: Path) -> Config:
#     config_json = json.loads(str(config_path))
#     return Config(
#         model_name=config_json["model_name"],
#         seed=config_json["seed"],
#         dataset=config_json["dataset"],
#         model=config_json["model"],
#         data_augmentation=config_json["data_augmentation"],
#         test_config=config_json["test_config"],
#         audio=config_json["audio"],
#     )
