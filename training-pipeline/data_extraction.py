import pandas as pd


def data_reading():
    # Extração de dados
    housing = pd.read_csv('training-pipeline/housing.csv')
    return housing
