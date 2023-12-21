from abc import ABC, abstractmethod
from typing import cast

import torch

from spira.adapter.config import FeatureEngineeringConfig
from spira.core.domain.audio import Audio, Audios, concatenate_audios
from spira.core.services.audio_processor import AudioProcessor


class AudioFeatureTransformer(ABC):
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor

    @abstractmethod
    def transform_into_features(self, audios: Audios) -> Audios:
        pass


class OverlappedAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(
        self, audio_processor: AudioProcessor, window_length: int, step_size: int
    ):
        super().__init__(audio_processor)
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
        super().__init__(audio_processor)

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

    def _add_padding_to_audio(self, audio: Audio, max_seq_len: int) -> Audio:
        padding_size = max_seq_len - audio.wav.size(0)
        zeros = torch.zeros(padding_size, audio.wav.size(1))
        padded_feature_wav = torch.cat([audio.wav, zeros], 0)
        return Audio(wav=padded_feature_wav, sample_rate=audio.sample_rate)


def create_audio_feature_transformer(
    audio_processor: AudioProcessor,
    feature_engineering_config: FeatureEngineeringConfig,
) -> AudioFeatureTransformer:
    if feature_engineering_config.split_wav_using_overlapping:
        return OverlappedAudioFeatureTransformer(
            audio_processor,
            feature_engineering_config.window_len,
            feature_engineering_config.step,
        )
    if feature_engineering_config.padding_with_max_length:
        return PaddedAudioFeatureTransformer(audio_processor)
    raise ValueError(f"Unknown feature engineering type")
