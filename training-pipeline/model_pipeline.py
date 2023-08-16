from data_extraction import data_reading
from data_split import training_and_test_set
from feature_engineering import extract_features
from hiperparameters_configuration import linear_regression
from train import fit
from validation import predict
from metrics import error_residual
from analytics import plot
import torch

# Extração de dados
dataset = data_reading()
print("Length: ", len(dataset))
example, label = dataset[0]
print("Features: ", example.shape)
print("Label: of index 0", label)

# Separação de conjunto de treino e de teste
train_size = int(len(dataset)*0.8)
test_size = len(dataset)-train_size
train_dataset, test_dataset = training_and_test_set(dataset, train_size, test_size)
print("{} examples for training and {} for testing".format(len(train_dataset), len(test_dataset)))


# # Feature engineering
# features_train = extract_features(X_train)
# features_test = extract_features(X_test)
#
# # Configuração de hiperparâmetros (linear regression tem zero)
# regr = linear_regression()
#
# # Treino
# regr = fit(regr, features_train, y_train)
#
# # Predição
# y_pred = predict(regr, features_test)
#
# # Métricas
# residuals = error_residual(y_pred, y_test)
#
# # Analytics
# plot(residuals)

