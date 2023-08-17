from data_extraction import *
from hiperparameters_configuration import *
from feature_engineering import *
from data_split import *
from train import *
from validation import *
from analytics import *


###### Data extraction ######
X, y = data_creation()

###### Feature engineering ######
X, y = feature_reshape(X, y)

###### Hyperparameters configuration #####
simple_1d_regression = Simple1DRegressionDataset(X, y)

###### train and test data split ######
training_loader = data_split(simple_1d_regression)

##### Train the model #####
model = fit(training_loader)

##### Validate the model #####
Y_pred = validate(model, X)

##### Analytics #####
plot(X, y, Y_pred)
