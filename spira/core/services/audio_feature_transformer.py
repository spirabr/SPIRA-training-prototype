from abc import ABC, abstractmethod

import torch
from spira.core.services.audio_processing import AudioProcessor

from spira.adapter.config import FeatureEngineeringConfig
from spira.core.domain.audio import Audio, concatenate_audios


class AudioFeatureTransformer(ABC):
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor

    @abstractmethod
    def transform_into_features(self, audios: list[Audio]) -> list[Audio]:
        pass


class OverlappedAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(
        self, audio_processor: AudioProcessor, window_length: int, step_size: int
    ):
        super().__init__(audio_processor)
        self.window_length = window_length
        self.step_size = step_size

    def transform_into_features(self, audios: list[Audio]) -> list[Audio]:
        return [self._transform_into_feature(audio) for audio in audios]

    def _transform_into_feature(self, audio: Audio) -> Audio:
        audio_slices = audio.create_slices(self.window_length, self.step_size)
        processed_audios = self.audio_processor.process_audios(audio_slices)
        return concatenate_audios(processed_audios)


class PaddedAudioFeatureTransformer(AudioFeatureTransformer):
    def __init__(self, audio_processor: AudioProcessor, hop_length: int):
        super().__init__(audio_processor)
        self.hop_length = hop_length

    def transform_into_features(self, audios: list[Audio]) -> list[Audio]:
        processed_audios = self.audio_processor.process_audios(audios)
        return self._add_padding_to_audios(processed_audios)

    def _add_padding_to_audios(self, audios: list[Audio]) -> list[Audio]:
        _, max_seq_length = self._calculate_min_max_audio_length(audios)
        return [self._add_padding_to_audio(audio, max_seq_length) for audio in audios]

    def _add_padding_to_audio(self, audio: Audio, max_seq_len: int) -> Audio:
        padding_size = max_seq_len - audio.wav.size(0)
        zeros = torch.zeros(padding_size, audio.wav.size(1))
        padded_feature_wav = torch.cat([audio.wav, zeros], 0)
        return Audio(wav=padded_feature_wav, sample_rate=audio.sample_rate)

    def _calculate_min_max_audio_length(self, audios: list[Audio]) -> tuple[int, int]:
        audio_lengths = [self._calculate_audio_length(audio) for audio in audios]
        return min(audio_lengths), max(audio_lengths)

    def _calculate_audio_length(self, audio: Audio):
        return int((audio.wav.shape[1] / self.hop_length) + 1)


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
        return PaddedAudioFeatureTransformer(
            audio_processor, feature_engineering_config.hop_length
        )
    raise ValueError(f"Unknown feature engineering type")
