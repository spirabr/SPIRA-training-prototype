from abc import ABC, abstractmethod
from functools import reduce
from typing import cast

import torch
from typing_extensions import Self

from spira.adapter.config import (
    FeatureEngineeringOptionsConfig,
    FeatureEngineeringParametersConfig,
)
from spira.adapter.random import Random
from spira.core.domain.audio import Audio, Audios, GeneratedAudio, concatenate_audios
from spira.core.domain.audio_processor import AudioProcessor
from spira.core.domain.wav import Wav


class AudioFeatureTransformer(ABC):
    @abstractmethod
    def transform_into_features(self, audios: Audios) -> Audios:
        pass


class NoisyAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(
        self,
        noises: Audios,
        noise_min_amp: float,
        noise_max_amp: float,
        num_samples: int,
        randomizer: Random,
    ):
        self.noises = noises
        self.noise_min_amp = noise_min_amp
        self.noise_max_amp = noise_max_amp
        self.num_samples = num_samples
        self.randomizer = randomizer

    def transform_into_features(self, audios: Audios) -> Audios:
        return audios.copy_using(
            [
                self._combine_audio_with_noise(audio, idx)
                for idx, audio in enumerate(audios)
            ]
        )

    def _combine_audio_with_noise(self, audio: Audio, extra_seed: int) -> Audio:
        noise_generator = self.create_noise_generator(extra_seed)
        return audio.combine_with_generated_audio(
            noise_generator._generate_noise(len(audio))
        )

    def create_noise_generator(self, extra_seed) -> Self:
        return cast(
            Self,
            NoisyAudioFeatureTransformer(
                self.noises,
                self.noise_min_amp,
                self.noise_max_amp,
                self.num_samples,
                self.randomizer.create_random(extra_seed),
            ),
        )

    def _generate_noise(self, limit_length: int) -> GeneratedAudio:
        noise_wav = self._generate_noise_wav(limit_length)
        return GeneratedAudio(wav=noise_wav)

    def _generate_noise_wav(self, limit_length: int) -> Wav:
        desired_amp = self.randomizer.get_random_float_in_interval(
            self.noise_min_amp, self.noise_max_amp
        )

        reshaped_noises = [
            self._reshape_audio(noise, desired_amp, limit_length)
            for noise in self.noises
        ]

        chosen_noises = self.randomizer.choose_n_elements(
            reshaped_noises, self.num_samples
        )

        noise = reduce(
            lambda noise, accumulated_noise: noise.wav + accumulated_noise.wav,
            chosen_noises,
        )

        return Wav(noise)

    def _reshape_audio(
        self, noise: Audio, desired_amp: float, limit_length: int
    ) -> Audio:
        min_length = min(len(noise), limit_length)
        max_length = max(len(noise), limit_length)

        desired_length = self.randomizer.get_randint_in_interval(min_length, max_length)
        return noise.rescale(desired_amp).resize(desired_length)


class OverlappedAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(
        self, audio_processor: AudioProcessor, window_length: int, step_size: int
    ):
        self.audio_processor = audio_processor
        self.window_length = window_length
        self.step_size = step_size

    def transform_into_features(self, audios: Audios) -> Audios:
        return audios.copy_using(
            [self._transform_into_feature(audio) for audio in audios]
        )

    def _transform_into_feature(self, audio: Audio) -> Audio:
        audio_slices = audio.create_slices(self.window_length, self.step_size)
        processed_audios = self.audio_processor.process_audios(audio_slices)
        return concatenate_audios(processed_audios)


class PaddedAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor

    def transform_into_features(self, audios: Audios) -> Audios:
        processed_audios: Audios = cast(
            Audios, self.audio_processor.process_audios(audios)
        )
        return self._add_padding_to_audios(processed_audios)

    def _add_padding_to_audios(self, audios: Audios) -> Audios:
        max_audio_length = audios.get_max_audio_length()
        return audios.copy_using(
            [self._add_padding_to_audio(audio, max_audio_length) for audio in audios]
        )

    @staticmethod
    def _add_padding_to_audio(audio: Audio, max_seq_len: int) -> Audio:
        padding_size = max_seq_len - audio.wav.size(0)
        zeros = torch.zeros(padding_size, audio.wav.size(1))
        padded_feature_wav = Wav(torch.cat([audio.wav, zeros], 0))
        return Audio(wav=padded_feature_wav, sample_rate=audio.sample_rate)


class MixedAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(self, randomizer: Random, alpha: float, beta: float):
        self.randomizer = randomizer
        self.alpha = alpha
        self.beta = beta

    def _mix_audios(self, first: Audio, second: Audio) -> tuple[Audio, Audio]:
        probability = self.randomizer.get_probability(self.alpha, self.beta)
        new_first = first * probability + second * (1 - probability)
        new_second = first * (1 - probability) + second * probability
        return new_first, new_second

    def transform_into_features(self, audios: Audios) -> Audios:
        mixed_audios: list[Audio] = []
        for first, second in audios.get_pairs_of_audios():
            mixed_audios.extend(self._mix_audios(first, second))

        return audios.copy_using(mixed_audios)


class AudioFeatureTransformerPipeline(AudioFeatureTransformer):
    def __init__(self, transformers: list[AudioFeatureTransformer]):
        self.transformers = transformers

    def transform_into_features(self, audios: Audios) -> Audios:
        for transformer in self.transformers:
            audios = transformer.transform_into_features(audios)
        return audios


def create_audio_feature_transformer(
    randomizer: Random,
    audio_processor: AudioProcessor,
    options: FeatureEngineeringOptionsConfig,
    parameters: FeatureEngineeringParametersConfig,
    num_samples: int,
    noises: Audios,
) -> AudioFeatureTransformerPipeline:
    transformers: list[AudioFeatureTransformer] = []

    if options.use_noise:
        transformers.append(
            NoisyAudioFeatureTransformer(
                noises,
                parameters.noisy_audio.noise_min_amp,
                parameters.noisy_audio.noise_max_amp,
                num_samples,
                randomizer,
            )
        )

    if options.use_overlapping:
        transformers.append(
            OverlappedAudioFeatureTransformer(
                audio_processor,
                parameters.overlapped_audio.window_length,
                parameters.overlapped_audio.step_size,
            )
        )

    if options.use_padding:
        transformers.append(PaddedAudioFeatureTransformer(audio_processor))

    if options.use_mixture:
        transformers.append(
            MixedAudioFeatureTransformer(
                randomizer, parameters.mixed_audio.alpha, parameters.mixed_audio.beta
            )
        )

    return AudioFeatureTransformerPipeline(transformers)
