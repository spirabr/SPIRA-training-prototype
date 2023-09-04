from pathlib import Path

from spira.training.config import load_spira_config
from spira.training.model import build_spira_model

###### Pre-config #####
config = load_spira_config(Path("../spira.json"))

###### Data extraction ######

###### train and test data split ######

###### Feature engineering ######


###### Hyperparameters configuration #####
model = build_spira_model(config)

##### Train the model #####

##### Validate the model #####

##### Analytics #####

