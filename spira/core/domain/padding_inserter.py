import torch

from spira.core.domain.audio import Audio


class PaddingGenerator:
    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len


    def add_padding_to_feature(self, feature: Audio) -> Audio:
        padding_size = self.max_seq_len - feature.wav.size(0)
        zeros = torch.zeros(padding_size, feature.wav.size(1))
        padded_feature_wav = torch.cat([feature.wav, zeros], 0)
        return Audio(wav=padded_feature_wav, sample_rate=feature.sample_rate)


    def add_padding_to_features(self, features: list[Audio]) -> list[Audio]:
        return [self.add_padding_to_feature(feature) for feature in features]
