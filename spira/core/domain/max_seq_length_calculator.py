from spira.adapter.config import AudioProcessorConfig
from spira.core.domain.audio import Audio


class AudiosLengthCalculator:
    def __init__(self, audio_processor_config: AudioProcessorConfig):
        self.hop_length = audio_processor_config.hop_length

    def calculate_min_max_audio_length(self, audios: list[Audio]) -> tuple[int, int]:
        audio_lengths = [self._calculate_audio_length(audio) for audio in audios]
        return min(audio_lengths), max(audio_lengths)

    def _calculate_audio_length(self, audio: Audio):
        return int((audio.wav.shape[1] / self.hop_length) + 1)
