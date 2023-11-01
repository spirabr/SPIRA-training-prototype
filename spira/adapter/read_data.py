from pathlib import Path

import pandas as pd


# todo: validar com renato se a mudança faz sentido
def read_patients_and_controls_list_csv(path: Path) -> [list[Path], list[Path]]:
    list_csv = pd.read_csv(path, sep=",").to_list
    patients = filter(lambda csv: csv[1] == "patient", list_csv)
    controls = filter(lambda csv: csv[1] == "control", list_csv)
    return patients, controls


def read_noises_list_csv(path: Path) -> [list[Path], list[Path]]:
    list_csv = pd.read_csv(path, sep=",").to_list
    noises_list = filter(lambda csv: csv[1] == "noises", list_csv)
    return noises_list
