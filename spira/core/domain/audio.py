import torch
from typing_extensions import Self


class Audio:
    def __init__(self, wav: torch.Tensor):
        self.wav = wav

    def combine_audio(self, audio: Self) -> Self:
        return Audio(wav=_combine_wavs(self.wav, audio.wav))

    def rescale_audio(self, desired_amplitude: float) -> Self:
        return Audio(rescale_wavs(self.wav, desired_amplitude))

    def resize_audio(self, desired_length: int) -> Self:
        return Audio(resize_wavs(self.wav, desired_length))


def resize_wavs(wav: torch.Tensor, length: int) -> torch.Tensor:
    return wav[0:length]


def rescale_wavs(wav: torch.Tensor, amplitude: float) -> torch.Tensor:
    reduce_factor = amplitude / float(wav.max())
    return wav * reduce_factor


def _combine_wavs(wav_1: torch.Tensor, wav_2: torch.Tensor) -> torch.Tensor:
    return wav_1 + wav_2
