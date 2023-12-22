from typing import NewType, cast

import torch
from torchaudio.transforms import Resample  # type: ignore

Wav = NewType("Wav", torch.Tensor)


def create_empty_wav() -> Wav:
    return Wav(torch.empty())


def resize_wav(wav: Wav, length: int) -> Wav:
    return Wav(wav[0:length])


def rescale_wav(wav: Wav, amplitude: float) -> Wav:
    return Wav(torch.mul(wav, amplitude / float(wav.max())))


def combine_wavs(wav_1: Wav, wav_2: Wav) -> Wav:
    return Wav(wav_1 + wav_2)


def resample_wav(wav: Wav, actual_sample_rate: int, desired_sample_rate: int) -> Wav:
    if desired_sample_rate == actual_sample_rate:
        return wav

    resample = Resample(actual_sample_rate, desired_sample_rate)
    return resample(wav)


def slice_wav(wav: Wav, start_index: int, end_index: int) -> Wav:
    return Wav(wav[start_index:end_index])


def concatenate_wavs(wavs: list[Wav]) -> Wav:
    return Wav(torch.cat(cast(list[torch.Tensor], wavs), dim=0))
