from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Extração de dados
housing = pd.read_csv('training-pipeline/housing.csv')

# Separação de conjunto de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(housing.median_income, housing.median_house_value, test_size = 0.2)


# Feature engineering
def extract_features(X):
    return np.array(X).reshape(-1, 1)


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

