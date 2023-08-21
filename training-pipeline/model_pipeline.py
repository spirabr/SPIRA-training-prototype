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
X_train_tensor = transform_inputs(X_train)
y_train_tensor = transform_label(y_train)
train_dataset = build_training_dataset(X_train_tensor, y_train_tensor)

###### Hyperparameters configuration #####
num_neurons = 10
model, loss_func = build_fnn(num_neurons)

##### Train the model #####
train_neural_network(model, loss_func, train_dataset, epochs=50)
# visualize2DSoftmax(X, y, model)

##### Validate the model #####
X_test_tensor = transform_inputs(X_test)
Y_pred = validate(model, X_test)

##### Analytics #####
plot(X_test, y_test, Y_pred, model)
