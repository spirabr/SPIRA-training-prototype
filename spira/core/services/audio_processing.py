from abc import abstractmethod
from enum import Enum

import torchaudio  # type: ignore
from torchaudio.transforms import MFCC, Resample  # type: ignore

from spira.adapter.config import AudioProcessorConfig
from spira.core.domain.audio import Audio


class AudioProcessorType(Enum):
    MFCC = "mfcc"
    SPECTROGRAM = "spectrogram"
    MELSPECTROGRAM = "melspectrogram"


class AudioProcessor(object):
    def __init__(self):
        self.transformer = self.create_transformer()

    @abstractmethod
    def create_transformer(self):
        pass

    def transform_into_feature(self, audio: Audio) -> Audio:
        # feature shape (Batch_size, n_features, timestamp)
        feature_wav = self.transformer(audio.wav)
        # transpose for (Batch_size, timestamp, n_features)
        transposed_feature_wav = feature_wav.transpose(1, 2)
        # remove batch dim = (timestamp, n_features)
        reshaped_feature_wav = transposed_feature_wav.reshape(
            transposed_feature_wav.shape[1:]
        )
        return Audio(wav=reshaped_feature_wav, sample_rate=audio.sample_rate)

    def transform_into_features(self, audios: list[Audio]) -> list[Audio]:
        return [self.transform_into_feature(audio) for audio in audios]


class MFCCAudioProcessor(AudioProcessor):
    def __init__(self, audio_config: AudioProcessorConfig):
        self.num_mels = audio_config.num_mels
        self.num_mfcc = audio_config.num_mfcc
        self.log_mels = audio_config.log_mels
        self.sample_rate = audio_config.sample_rate
        self.hop_length = audio_config.hop_length
        self.win_length = audio_config.win_length
        self.n_ftt = audio_config.n_fft

    def create_transformer(self):
        return MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.num_mfcc,
            log_mels=self.log_mels,
            melkwargs={
                "n_fft": self.n_fft,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
                "n_mels": self.num_mels,
            },
        )


class SpectrogramAudioProcessor(AudioProcessor):
    def create_transformer(self):
        pass


class MelspectrogramAudioProcessor(AudioProcessor):
    def create_transformer(self):
        pass


def create_audio_processor(audio_config: AudioProcessorConfig):
    match audio_config.feature_type:
        case AudioProcessorType.MFCC:
            return MFCCAudioProcessor(audio_config)
        case AudioProcessorType.SPECTROGRAM:
            return SpectrogramAudioProcessor()
        case AudioProcessorType.MELSPECTROGRAM:
            return MelspectrogramAudioProcessor()
