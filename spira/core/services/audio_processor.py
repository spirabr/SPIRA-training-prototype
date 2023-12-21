from abc import abstractmethod
from enum import Enum

import torchaudio  # type: ignore
from torchaudio.transforms import MFCC, Resample  # type: ignore

from spira.adapter.config import (
    AudioProcessorConfig,
    MFCCAudioProcessorConfig,
    MelspectrogramAudioProcessorConfig,
    SpectrogramAudioProcessorConfig,
)
from spira.core.domain.audio import Audio, Audios, GeneratedAudios


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

    def process_audio(self, audio: Audio) -> Audio:
        # feature shape (Batch_size, n_features, timestamp)
        feature_wav = self.transformer(audio.wav)
        # transpose for (Batch_size, timestamp, n_features)
        transposed_feature_wav = feature_wav.transpose(1, 2)
        # remove batch dim = (timestamp, n_features)
        reshaped_feature_wav = transposed_feature_wav.reshape(
            transposed_feature_wav.shape[1:]
        )
        return Audio(wav=reshaped_feature_wav, sample_rate=audio.sample_rate)

    def process_audios(
        self, audios: Audios | GeneratedAudios
    ) -> Audios | GeneratedAudios:
        return audios.copy_using([self.process_audio(audio) for audio in audios])


class MFCCAudioProcessor(AudioProcessor):
    def __init__(self, config: MFCCAudioProcessorConfig):
        super().__init__()
        self.config = config

    def create_transformer(self):
        return MFCC(
            sample_rate=self.config.sample_rate,
            n_mfcc=self.config.num_mfcc,
            log_mels=self.config.log_mels,
            melkwargs={
                "n_fft": self.config.n_fft,
                "win_length": self.config.win_length,
                "hop_length": self.config.hop_length,
                "n_mels": self.config.num_mels,
            },
        )


class SpectrogramAudioProcessor(AudioProcessor):
    def __init__(self, config: SpectrogramAudioProcessorConfig):
        super().__init__()
        self.config = config

    def create_transformer(self):
        pass


class MelspectrogramAudioProcessor(AudioProcessor):
    def __init__(self, config: MelspectrogramAudioProcessorConfig):
        super().__init__()
        self.config = config

    def create_transformer(self):
        pass


def create_audio_processor(config: AudioProcessorConfig):
    match config.feature_type:
        case AudioProcessorType.MFCC:
            return MFCCAudioProcessor(config.mfcc)
        case AudioProcessorType.SPECTROGRAM:
            return SpectrogramAudioProcessor(config.spectrogram)
        case AudioProcessorType.MELSPECTROGRAM:
            return MelspectrogramAudioProcessor(config.melspectrogram)
