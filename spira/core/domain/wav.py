import torch
from torchaudio.transforms import Resample


def create_empty_wav() -> torch.Tensor:
    return torch.Tensor()


def resize_wav(wav: torch.Tensor, length: int) -> torch.Tensor:
    return wav[0:length]


def rescale_wav(wav: torch.Tensor, amplitude: float) -> torch.Tensor:
    return torch.mul(wav, amplitude / float(wav.max()))


def combine_wavs(wav_1: torch.Tensor, wav_2: torch.Tensor) -> torch.Tensor:
    return wav_1 + wav_2


def resample_wav(
    wav: torch.Tensor, actual_sample_rate: int, desired_sample_rate: int
) -> torch.Tensor:
    if desired_sample_rate == actual_sample_rate:
        return wav

    resample = Resample(actual_sample_rate, desired_sample_rate)
    return resample(wav)


def slice_wav(wav: torch.Tensor, start_index: int, end_index: int) -> torch.Tensor:
    return wav[start_index:end_index]


def concatenate_wavs(wavs: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(wavs, dim=0)
