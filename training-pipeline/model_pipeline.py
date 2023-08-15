from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import matplotlib.pyplot as plt

from data_extraction import data_reading
from data_split import training_and_test_set
from feature_engineering import extract_features
from hiperparameters_configuration import

# Extração de dados
housing = data_reading()

# Separação de conjunto de treino e de teste
X_train, X_test, y_train, y_test = training_and_test_set(housing)


# Feature engineering
features_train = extract_features(X_train)
features_test = extract_features(X_test)

# Configuração de hiperparâmetros (linear regression tem zero)
regr = LinearRegression()

# Treino
regr.fit(features_train, y_train)

# Predição
y_pred = regr.predict(features_test)

# Métricas
residuals = y_pred - y_test

# Analytics
plt.hist(residuals)
plt.savefig('training-pipeline/plot.png')  # Save the plot as an image file

