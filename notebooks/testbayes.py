import torch

from grape.datasets import BradleyDoublePlus

from grape.models import SimpleGNN, MGConv
from torch import nn
import torch

data = BradleyDoublePlus(split_type='random')

node_hidden_dim = 128
batch_size = 32

#model_message = MPNN(num_layers=2, edge_hidden_dim=128, node_hidden_dim=node_hidden_dim,
#                node_in_dim=data.num_node_features, edge_in_dim=data.num_edge_features,
#                num_gru_layers=1)

model_message = MGConv(edge_hidden_dim=128, node_hidden_dim=node_hidden_dim, node_in_dim=data.num_node_features, edge_in_dim=data.num_edge_features)

out_model = nn.Sequential(
    nn.Linear(node_hidden_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)


model = SimpleGNN(model_message=model_message,
                  out_model=out_model)

train_data, val_data, test_data = data.train, data.val, data.test

from grape.utils import RayTuner
from grape.optim import adam_objective, adam_default_search_space
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray import tune, train
from functools import partial

objective = adam_objective
objective = partial(objective, train_loader=train_data, val_loader=val_data, model=model)
search_space = adam_default_search_space()
search_algo = HyperOptSearch()
print(search_space)

#tuner = tune.Tuner(
#            objective,
#            tune_config=tune.TuneConfig(
#                metric='mean_squared_error',
#                mode='min',
#                search_alg=search_algo,
#            ),
#            run_config=train.RunConfig(
#                stop={"training_iteration": 5},
#            ),
#            param_space=search_space,
#)
#results = tuner.fit()
#print(results.get_best_result().config)


optimizer = RayTuner(objective=adam_objective, search_space=search_space,train_loader=train_data, val_loader=val_data, model=model)
optimizer.fit()
