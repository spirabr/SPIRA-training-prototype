from typing import cast

import torch
from typing_extensions import Self


class Audio:
    def __init__(self, wav: torch.Tensor):
        self.wav = wav

    def __len__(self):
        return len(self.wav)

    def combine_audio(self, audio: Self) -> Self:
        return cast(Self, Audio(wav=_combine_wavs(self.wav, audio.wav)))

    def rescale_audio(self, desired_amplitude: float) -> Self:
        return cast(Self, Audio(wav=_rescale_wav(self.wav, desired_amplitude)))

    def resize_audio(self, desired_length: int) -> Self:
        return cast(Self, Audio(wav=_resize_wav(self.wav, desired_length)))


def _resize_wav(wav: torch.Tensor, length: int) -> torch.Tensor:
    return wav[0:length]


def _rescale_wav(wav: torch.Tensor, amplitude: float) -> torch.Tensor:
    return torch.mul(wav, amplitude / float(wav.max()))


def _combine_wavs(wav_1: torch.Tensor, wav_2: torch.Tensor) -> torch.Tensor:
    return wav_1 + wav_2
