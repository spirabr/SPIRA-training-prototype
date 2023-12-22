import math
from typing import Callable, Iterable, Iterator, Optional, cast

import torchaudio  # type: ignore
from typing_extensions import Self

from spira.adapter.valid_path import ValidPath
from spira.core.domain.wav import (
    Wav,
    combine_wavs,
    concatenate_wavs,
    create_empty_wav,
    resample_wav,
    rescale_wav,
    resize_wav,
    slice_wav,
)


class GeneratedAudio:
    def __init__(self, wav: Wav):
        self.wav = wav


class GeneratedAudios:
    def __init__(self, audios: list["Audio"]):
        self.audios = audios

    def __len__(self):
        return len(self.audios)

    def __iter__(self):
        return iter(self.audios)

    @staticmethod
    def copy_using(audios: list["Audio"]):
        return GeneratedAudios(audios)


class Audio:
    EMPTY_SAMPLE_RATE = 0

    def __init__(self, wav: Wav, sample_rate: int):
        self.wav = wav
        self.sample_rate = sample_rate

    @classmethod
    def load(cls, path: ValidPath, normalize: bool) -> Self:
        wav, sample_rate = torchaudio.load(str(path), normalize=normalize)  # noqa
        return cast(Self, Audio(wav=wav, sample_rate=sample_rate))

    @classmethod
    def create_empty(cls):
        return cast(
            Self, Audio(wav=create_empty_wav(), sample_rate=Audio.EMPTY_SAMPLE_RATE)
        )

    def __len__(self):
        return math.ceil(len(self.wav) / self.sample_rate)

    def __add__(self, other: Self | float) -> Self:
        term = other.wav if isinstance(other, Audio) else other
        return cast(Self, Audio(Wav(self.wav + term), self.sample_rate))

    def __mul__(self, other: Self | float) -> Self:
        term = other.wav if isinstance(other, Audio) else other
        return cast(Self, Audio(Wav(self.wav * term), self.sample_rate))

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

    def create_slices(self, window_length: int, step_size: int) -> GeneratedAudios:
        slicer = Audio.Slicer(self, window_length, step_size)
        return cast(GeneratedAudios, list(slicer))

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


class Audios:
    _min_audio_length: Optional[int] = None
    _max_audio_length: Optional[int] = None

    def __init__(self, audios: list[Audio], hop_length: int):
        self.audios = audios
        self.hop_length = hop_length

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, index: int) -> Audio:
        return self.audios[index]

    def __iter__(self) -> Iterator[Audio]:
        return iter(self.audios)

    def __add__(self, other: Self) -> Self:
        assert self.hop_length == other.hop_length, "Hop lengths are not equal"
        return self.copy_using(self.audios + other.audios)

    def get_pairs_of_audios(self) -> Iterable[tuple[Audio, Audio]]:
        iterator = iter(self.audios)
        while iterator:
            yield next(iterator), next(iterator)

    @classmethod
    def load(cls, paths: list[ValidPath], hop_length: int, normalize: bool) -> Self:
        return cast(
            Self, Audios([Audio.load(path, normalize) for path in paths], hop_length)
        )

    def get_max_audio_length(self) -> int:
        self._lazy_calculate_min_max_audio_length()
        return cast(int, self._max_audio_length)

    def get_min_audio_length(self) -> int:
        self._lazy_calculate_min_max_audio_length()
        return cast(int, self._min_audio_length)

    def _lazy_calculate_min_max_audio_length(self):
        if self._min_audio_length is None or self._max_audio_length is None:
            self._calculate_min_max_audio_length()

    def _calculate_min_max_audio_length(self):
        audio_lengths = [self._calculate_audio_length(audio) for audio in self.audios]
        self._min_audio_length = min(audio_lengths)
        self._max_audio_length = max(audio_lengths)
        assert self._min_audio_length is not None and self._max_audio_length is not None

    def _calculate_audio_length(self, audio: Audio):
        return int((audio.wav.shape[1] / self.hop_length) + 1)

    def apply(self, transformer: Callable[[Audio, int], Audio]) -> Self:
        return cast(
            Self,
            Audios(
                [transformer(audio, idx) for idx, audio in enumerate(self.audios)],
                self.hop_length,
            ),
        )

    def copy_using(self, audios: list[Audio]) -> Self:
        return cast(Self, Audios(audios, self.hop_length))


def concatenate_audios(audios: Audios | GeneratedAudios) -> Audio:
    if len(audios) == 0:
        return Audio(wav=create_empty_wav(), sample_rate=0)

    concatenated_wav = concatenate_wavs(get_wavs_from_audios(audios))

    if _check_audios_have_same_sample_rate(audios):
        raise ValueError("Sample rates are not equal")

    return cast(
        Audio, Audio(wav=concatenated_wav, sample_rate=audios.audios[0].sample_rate)
    )


def get_wavs_from_audios(audios: Audios | GeneratedAudios) -> list[Wav]:
    return [audio.wav for audio in audios]


def _get_sample_rates_from_audios(audios: Audios | GeneratedAudios) -> list[int]:
    return [audio.sample_rate for audio in audios]


def _check_audios_have_same_sample_rate(audios: Audios | GeneratedAudios) -> bool:
    return len(set(_get_sample_rates_from_audios(audios))) > 1
