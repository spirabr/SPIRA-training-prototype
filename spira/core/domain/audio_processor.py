from abc import abstractmethod
from enum import Enum

from torchaudio.transforms import MFCC  # type: ignore

from spira.adapter.config import (
    AudioProcessorParametersConfig,
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
    def __init__(self, hop_length: int):
        self.hop_length = hop_length
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
    def __init__(self, config: MFCCAudioProcessorConfig, hop_length: int):
        super().__init__(hop_length)
        self.config = config

    def create_transformer(self):
        return MFCC(
            sample_rate=self.config.sample_rate,
            n_mfcc=self.config.num_mfcc,
            log_mels=self.config.log_mels,
            melkwargs={
                "n_fft": self.config.n_fft,
                "win_length": self.config.win_length,
                "hop_length": self.hop_length,
                "n_mels": self.config.num_mels,
            },
        )


class SpectrogramAudioProcessor(AudioProcessor):
    def __init__(self, config: SpectrogramAudioProcessorConfig, hop_length: int):
        super().__init__(hop_length)
        self.config = config

    def create_transformer(self):
        pass


class MelspectrogramAudioProcessor(AudioProcessor):
    def __init__(self, config: MelspectrogramAudioProcessorConfig, hop_length: int):
        super().__init__(hop_length)
        self.config = config

    def create_transformer(self):
        pass


def create_audio_processor(parameters: AudioProcessorParametersConfig):
    match parameters.feature_type:
        case AudioProcessorType.MFCC:
            return MFCCAudioProcessor(
                parameters.mfcc,
                parameters.hop_length,
            )
        case AudioProcessorType.SPECTROGRAM:
            return SpectrogramAudioProcessor(
                parameters.spectrogram,
                parameters.hop_length,
            )
        case AudioProcessorType.MELSPECTROGRAM:
            return MelspectrogramAudioProcessor(
                parameters.melspectrogram,
                parameters.hop_length,
            )
