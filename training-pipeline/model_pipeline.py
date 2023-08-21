from data_extraction import *
from hiperparameters_configuration import *
from feature_engineering import *
from data_split import *
from train import *
from validation import *
from analytics import *


###### Data extraction ######
X, y = generate_data()

###### train and test data split ######
X_train, X_test, y_train, y_test = split_data(X, y)

###### Feature engineering ######
train_features = extract_features(X_train, y_train)
test_features = extract_features(X_test, y_test)

###### Hyperparameters configuration #####
num_neurons = 10
model, loss_func = build_fnn(num_neurons)

##### Train the model #####
train_neural_network(model, loss_func, train_features, epochs=200)

##### Validate the model #####
Y_pred = validate(model, test_features)

##### Analytics #####
plot(X_test, y_test, Y_pred)
