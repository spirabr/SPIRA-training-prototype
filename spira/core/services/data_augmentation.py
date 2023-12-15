from spira.core.domain.audio import Audio
from spira.core.domain.noise_generator import NoiseGenerator


class NoiseApplier:
    def __init__(self, num_samples: int, noise_generator: NoiseGenerator):
        self.num_samples = num_samples
        self.noise_generator = noise_generator

    def generate_noisy_audios(self, audios: list[Audio]) -> list[Audio]:
        return [
            self._combine_audio_with_noise(patient, idx)
            for idx, patient in enumerate(audios)
        ]

    def _combine_audio_with_noise(self, audio: Audio, extra_seed: int):
        noise_generator = self.noise_generator.create_noise_generator(extra_seed)
        return audio.combine(
            noise_generator.generate_noise(self.num_samples, len(audio))
        )
