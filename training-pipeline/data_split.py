import torch.utils.data


# Separação de conjunto de treino e de teste
def training_and_test_set(dataset, train_size, test_size):
    return torch.utils.data.random_split(dataset, (train_size, test_size))
