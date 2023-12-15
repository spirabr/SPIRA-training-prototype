from typing import Optional, cast

import torch
import torchaudio  # type: ignore
from torchaudio.transforms import MFCC, Resample  # type: ignore
from typing_extensions import Self

from spira.adapter.valid_path import ValidPath


class Audio:
    def __init__(self, wav: torch.Tensor, sample_rate: Optional[int]):
        self.wav = wav
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.wav)

    def combine(self, audio: Self) -> Self:
        combined_sample_rate = self._combine_sample_rate(audio.sample_rate)

        return cast(
            Self,
            Audio(
                wav=_combine_wavs(self.wav, audio.wav),
                sample_rate=combined_sample_rate,
            ),
        )

    def rescale(self, desired_amplitude: float) -> Self:
        return cast(
            Self,
            Audio(
                wav=_rescale_wav(self.wav, desired_amplitude),
                sample_rate=self.sample_rate,
            ),
        )

    def resize(self, desired_length: int) -> Self:
        return cast(
            Self,
            Audio(
                wav=_resize_wav(self.wav, desired_length),
                sample_rate=self.sample_rate,
            ),
        )

    def resample(self, desired_sample_rate: int) -> Self:
        return cast(
            Self,
            Audio(
                wav=_resample_wav(self.wav, self.sample_rate, desired_sample_rate),
                sample_rate=desired_sample_rate,
            ),
        )

    def _combine_sample_rate(self, other_sample_rate: Optional[int]):
        # TODO: Sample rate should always exist!
        # This is a workaround because noises don't have a sample rate
        if self.sample_rate:
            return self.sample_rate
        if other_sample_rate:
            return other_sample_rate
        raise ValueError("One of the waves should have their sample rate!")

    @classmethod
    def load(cls, path: ValidPath, normalize: bool) -> Self:
        wav, sample_rate = torchaudio.load(str(path), normalize=normalize)
        return cast(Self, Audio(wav=wav, sample_rate=sample_rate))


def _resize_wav(wav: torch.Tensor, length: int) -> torch.Tensor:
    return wav[0:length]


def _rescale_wav(wav: torch.Tensor, amplitude: float) -> torch.Tensor:
    return torch.mul(wav, amplitude / float(wav.max()))


def _combine_wavs(wav_1: torch.Tensor, wav_2: torch.Tensor) -> torch.Tensor:
    return wav_1 + wav_2


def _resample_wav(
    wav: torch.Tensor, actual_sample_rate: int, desired_sample_rate: int
) -> torch.Tensor:
    if desired_sample_rate == actual_sample_rate:
        return wav

    resample = Resample(actual_sample_rate, desired_sample_rate)
    return resample(wav)


def load_audios(paths: list[ValidPath], normalize: bool) -> list[Audio]:
    return [Audio.load(path, normalize) for path in paths]
