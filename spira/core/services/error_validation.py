import os

from spira.adapter.config import Config


# todo: Fazer as próprias exceções ao invés de RunTimeError


def validate_config_is_valid(config: Config):
    if (
        config.dataset.padding_with_max_length
        ^ config.dataset.split_wav_using_overlapping
    ):
        RuntimeError(
            "You cannot use the padding_with_max_length option in conjunction with the split_wav_using_overlapping option, disable one of them !!"
        )


def validate_files_exist(config: Config):
    if os.path.isfile(config.dataset.noise_csv):
        FileExistsError("Noise CSV file don't exists! Fix it in config.json")
    if os.path.isfile(config.dataset.test_csv):
        FileExistsError("Test or Train CSV file don't exists! Fix it in config.json")
