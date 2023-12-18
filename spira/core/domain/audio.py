import math
from typing import Iterable, cast

import torch
import torchaudio  # type: ignore
from torchaudio.transforms import MFCC, Resample  # type: ignore
from typing_extensions import Self

from spira.adapter.valid_path import ValidPath
from spira.core.domain.wav import (
    combine_wavs,
    concatenate_wavs,
    create_empty_wav,
    resample_wav,
    rescale_wav,
    resize_wav,
    slice_wav,
)


class GeneratedAudio:
    def __init__(self, wav: torch.Tensor):
        self.wav = wav


class Audio:
    def __init__(self, wav: torch.Tensor, sample_rate: int):
        self.wav = wav
        self.sample_rate = sample_rate

    @classmethod
    def load(cls, path: ValidPath, normalize: bool) -> Self:
        wav, sample_rate = torchaudio.load(str(path), normalize=normalize)  # noqa
        return cast(Self, Audio(wav=wav, sample_rate=sample_rate))

    @classmethod
    def create_empty(cls):
        EMPTY_SAMPLE_RATE = 0
        return cast(Self, Audio(wav=create_empty_wav(), sample_rate=EMPTY_SAMPLE_RATE))

    def __len__(self):
        return math.ceil(len(self.wav) / self.sample_rate)

    def combine(self, audio: Self) -> Self:
        if self.sample_rate != audio.sample_rate:
            raise ValueError("Sample rates should be the same!")

        return cast(
            Self,
            Audio(
                wav=combine_wavs(self.wav, audio.wav),
                sample_rate=self.sample_rate,
            ),
        )

    def rescale(self, desired_amplitude: float) -> Self:
        return cast(
            Self,
            Audio(
                wav=rescale_wav(self.wav, desired_amplitude),
                sample_rate=self.sample_rate,
            ),
        )

    def resize(self, desired_length: int) -> Self:
        return cast(
            Self,
            Audio(
                wav=resize_wav(self.wav, desired_length),
                sample_rate=self.sample_rate,
            ),
        )

    def resample(self, desired_sample_rate: int) -> Self:
        return cast(
            Self,
            Audio(
                wav=resample_wav(self.wav, self.sample_rate, desired_sample_rate),
                sample_rate=desired_sample_rate,
            ),
        )

    def create_slices(self, window_length: int, step_size: int) -> list[Self]:
        slicer = Audio.Slicer(self, window_length, step_size)
        return cast(list[Self], list(slicer))

    def create_slice(self, start_index: int, end_index: int) -> Self:
        if start_index < 0 or end_index < 0 or start_index >= end_index:
            raise ValueError(f"Invalid range [{start_index}:{end_index}]")

        return cast(
            Self,
            Audio(
                wav=slice_wav(
                    wav=self.wav,
                    # Audios are indexed in sample_rate chunks
                    start_index=start_index * self.sample_rate,
                    end_index=end_index * self.sample_rate,
                ),
                sample_rate=self.sample_rate,
            ),
        )

    def combine_with_generated_audio(self, generated_audio: GeneratedAudio) -> Self:
        return cast(
            Self,
            Audio(
                wav=combine_wavs(self.wav, generated_audio.wav),
                sample_rate=self.sample_rate,
            ),
        )

    class Slicer(Iterable[Self]):
        def __init__(self, audio: "Audio", window_length: int, step_size: int):
            self.audio = audio
            self.window_length = window_length
            self.step_size = step_size

            self.slice_index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.slice_index >= len(self.audio):
                raise StopIteration

            start = self.slice_index
            end = self.slice_index + self.window_length

            sliced_audio = self.audio.create_slice(start, end)
            self.slice_index += self.step_size

            return sliced_audio


def load_audios(paths: list[ValidPath], normalize: bool) -> list[Audio]:
    return [Audio.load(path, normalize) for path in paths]


def concatenate_audios(audios: list[Audio]) -> Audio:
    if len(audios) == 0:
        return Audio(wav=torch.Tensor(), sample_rate=0)

    concatenated_wav = concatenate_wavs(get_wavs_from_audios(audios))

    if _check_audios_have_same_sample_rate(audios):
        raise ValueError(f"Sample rates are not equal")

    return Audio(wav=concatenated_wav, sample_rate=audios[0].sample_rate)


def get_wavs_from_audios(audios: list[Audio]) -> list[torch.Tensor]:
    return [audio.wav for audio in audios]


def _get_sample_rates_from_audios(audios: list[Audio]) -> list[int]:
    return [audio.sample_rate for audio in audios]


def _check_audios_have_same_sample_rate(audios: list[Audio]) -> bool:
    return len(set(_get_sample_rates_from_audios(audios))) > 1
