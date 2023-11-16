from spira.core.domain.audio import Audio
from spira.core.domain.noise_generator import NoiseGenerator


def generate_noisy_audios(
    audios: list[Audio], num_noise: int, base_noise_generator: NoiseGenerator
) -> list[Audio]:
    def combine_audio_with_noise(patient: Audio, seed: int):
        return _combine_audios_with_noise(
            patient, seed, base_noise_generator, num_noise
        )

    return [
        combine_audio_with_noise(patient, idx) for idx, patient in enumerate(audios)
    ]


def _combine_audios_with_noise(
    audio: Audio,
    extra_seed: int,
    base_noise_generator: NoiseGenerator,
    num_samples: int,
):
    noise_generator = base_noise_generator.create_noise_generator(extra_seed)
    return audio.combine_audio(noise_generator.generate_noise(num_samples, len(audio)))
