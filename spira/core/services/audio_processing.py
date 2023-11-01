from pathlib import Path

import torchaudio  # type: ignore
from torchaudio.transforms import MFCC, Resample  # type: ignore

from spira.adapter.config import ConfigAudio
from spira.core.domain.audio import Audio

"""
Responsible to process the audio input, 

"""


class AudioProcessor(object):
    def __init__(self, config_audio: ConfigAudio):
        self.feature = config_audio.feature
        self.num_mels = config_audio.num_mels
        self.num_mfcc = config_audio.num_mfcc
        self.log_mels = config_audio.log_mels
        self.mel_fmin = config_audio.mel_fmin
        self.mel_fmax = config_audio.mel_fmax
        self.normalize = config_audio.normalize
        self.sample_rate = config_audio.sample_rate
        self.n_fft = config_audio.n_fft
        self.hop_length = config_audio.hop_length
        self.win_length = config_audio.win_length

        valid_features = ["spectrogram", "melspectrogram", "mfcc"]
        if self.feature not in valid_features:
            raise ValueError("Invalid Feature: " + str(self.feature))

    # Can we remove support to Spectogram and Melspectogram?
    def wav2feature(self, y):
        audio_class = MFCC(
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

        return audio_class(y)

    # Are these methods needed?
    def get_feature_from_audio_path(self, audio_path):
        return self.wav2feature(self.load_audio(audio_path))

    def get_feature_from_audio(self, wav):
        return self.wav2feature(wav)

    def load_audio(self, path: Path) -> Audio:
        wav, sample_rate = torchaudio.load(path, normalize=self.normalize)

        # resample audio for specific samplerate
        if sample_rate != self.sample_rate:
            resample = Resample(sample_rate, self.sample_rate)
            wav = resample(wav)

        return Audio(wav)

    # todo: Validar com renato se a mudança faz sentido
    def load_audio_from_list(self, path_list: list[Path]) -> list[Audio]:
        return [self.load_audio(file) for file in path_list]
