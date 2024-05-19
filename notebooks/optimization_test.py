############################ Optimization script ############################################

from ray.air import RunConfig

from grape.models import AFP
from grape.utils import EarlyStopping, train_model
from grape.datasets import FreeSolv
from functools import partial
import torch
from torch.optim import lr_scheduler
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune import Tuner
from ray import tune

import ConfigSpace as CS

from ray import train, tune

data = FreeSolv(split_type='random')

model_name = 'AFP'


def trainable(config: dict, dataset):
    model = AFP(node_in_dim=44, edge_in_dim=12, num_layers_mol=config["depth"],
                hidden_dim=config["gnn_hidden_dim"],)

    train_set, val_set, test_set = dataset.train, dataset.val, dataset.test

    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=1e-6)
    early_Stopper = EarlyStopping(patience=20, model_name='random', skip_save=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, min_lr=0.0000000000001, patience=30)

    train_loss, val_loss = train_model(model=model,
                                       loss_func='mse',
                                       optimizer=optimizer,
                                       train_data_loader=train_set,
                                       val_data_loader=val_set,
                                       batch_size=32,
                                       epochs=500,
                                       early_stopper=early_Stopper,
                                       scheduler=scheduler)

    best_loss = early_Stopper.best_score

    train.report({"mse_loss": best_loss})  # This sends the score to Tune.


################################# Search space ######################################

config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(
    CS.UniformIntegerHyperparameter("depth", lower=1, upper=5))
config_space.add_hyperparameter(
    CS.UniformIntegerHyperparameter("gnn_hidden_dim", lower=32, upper=256))
config_space.add_hyperparameter(
    CS.UniformFloatHyperparameter('initial_lr', lower=1e-5, upper=1e-1, log=True))
# config_space.add_hyperparameter(
#    CS.UniformFloatHyperparameter("height", lower=-100, upper=100))
# config_space.add_hyperparameter(
#    CS.CategoricalHyperparameter(
#        name="activation", choices=["relu", "tanh"]))

################################# -------------- ######################################


### Define search algorithm
algo = TuneBOHB(config_space, metric="mse_loss", mode="min")

### Get the trial control algorithm
bohb = HyperBandForBOHB(
    time_attr="training_iteration",
    metric="mse_loss",
    mode="min",
    max_t=100)

### initialize the trainable with the dataset from the top
my_trainable = partial(trainable, dataset=data)

### Initialize the tuner
tuner = Tuner(my_trainable, tune_config=tune.TuneConfig(scheduler=HyperBandForBOHB(),
                                        search_alg=algo,
                                        mode='min',
                                        metric="mse_loss"))

result = tuner.fit()

import json

best_result = result.get_best_result(metric="mse_loss", mode="min")
best_config = best_result.config
best_metrics = best_result.metrics

results_to_save = {
    "best_config": best_config,
    "best_metrics": best_metrics
}

file_name = "./results/best_hyperparameters_" + model_name + ".json"

with open(file_name, "w") as file:
    json.dump(results_to_save, file, indent=4)

#with open('./results/AFP_optimization_results.txt','a') as file:
#    string = result.get_best_result(metric="mse_loss", mode="min") + '\n'
#    file.write(string)
