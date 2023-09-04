import json
from pathlib import Path

from pydantic import BaseModel
from pydantic.dataclasses import dataclass


@dataclass
class SpiraConfigDataset:
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
class SpiraConfigModel:
    fc1_dim: int
    fc2_dim: int


@dataclass
class SpiraConfigDataAugmentation:
    insert_noise: bool
    num_noise_control: int
    num_noise_patient: int
    noise_max_amp: float
    noise_min_amp: float


@dataclass
class SpiraConfigTestConfig:
    batch_size: int
    num_workers: int


@dataclass
class SpiraConfigAudio:
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


class SpiraConfig(BaseModel):
    model_name: str
    seed: int
    dataset: SpiraConfigDataset
    model: SpiraConfigModel
    data_augmentation: SpiraConfigDataAugmentation
    test_config: SpiraConfigTestConfig
    audio: SpiraConfigAudio

    def num_features(self):
        if self.audio.feature == 'spectrogram':
            return self.audio.num_freq
        elif self.audio.feature == 'melspectrogram':
            return self.audio.num_mels
        elif self.audio.feature == 'mfcc':
            return self.audio.num_mfcc
        return None


def load_spira_config(config_path: Path) -> SpiraConfig:
    config_json = json.loads(str(config_path))
    return SpiraConfig(config_json)
