from data_extraction import data_reading
from data_split import training_and_test_set
from feature_engineering import extract_features
from hiperparameters_configuration import linear_regression
from train import fit
from prediction import predict
from metrics import error_residual
from analytics import plot

# Extração de dados
housing = data_reading()

# Separação de conjunto de treino e de teste
X_train, X_test, y_train, y_test = training_and_test_set(housing)

# Feature engineering
features_train = extract_features(X_train)
features_test = extract_features(X_test)

# Configuração de hiperparâmetros (linear regression tem zero)
regr = linear_regression()

# Treino
regr = fit(regr, features_train, y_train)

# Predição
y_pred = predict(regr, features_test)

# Métricas
residuals = error_residual(y_pred, y_test)

# Analytics
plot(residuals)

