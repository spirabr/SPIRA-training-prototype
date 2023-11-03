from functools import reduce

import torch

from spira.adapter.random import Random
from spira.core.domain.audio import Audio


class NoiseGenerator:
    def __init__(
        self,
        noises: list[Audio],
        noise_min_amp: float,
        noise_max_amp: float,
        randomizer: Random,
    ):
        self.noises = noises
        self.noise_min_amp = noise_min_amp
        self.noise_max_amp = noise_max_amp
        self.randomizer = randomizer

    def create_noise_generator(self, extra_seed):
        return NoiseGenerator(
            self.noises,
            self.noise_min_amp,
            self.noise_max_amp,
            self.randomizer.create_random(extra_seed),
        )

    def generate_noise(self, num_samples: int, limit_length: int) -> Audio:
        return Audio(wav=self._generate_noise_wav(num_samples, limit_length))

    def _generate_noise_wav(self, num_samples: int, limit_length: int) -> torch.Tensor:
        desired_amp = self.randomizer.get_random_float_in_interval(
            self.noise_min_amp, self.noise_max_amp
        )

        reshaped_noises = [
            self._reshape_audio(noise, desired_amp, limit_length)
            for noise in self.noises
        ]

        chosen_noises = self.randomizer.choose_n_elements(reshaped_noises, num_samples)

        return reduce(
            lambda noise, accumulated_noise: noise.wav + accumulated_noise.wav, chosen_noises
        )

    def _reshape_audio(
        self, noise: Audio, desired_amp: float, limit_length: int
    ) -> Audio:
        min_length = min(len(noise), limit_length)
        max_length = max(len(noise), limit_length)

        desired_length = self.randomizer.get_randint_in_interval(min_length, max_length)
        return noise.rescale_audio(desired_amp).resize_audio(desired_length)
