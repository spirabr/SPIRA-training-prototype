from sklearn.model_selection import train_test_split


# Separação de conjunto de treino e de teste
def training_and_test_set(housing):
    return train_test_split(housing.median_income, housing.median_house_value, test_size=0.2)