# Inspired by the tutorial at https://docs.ray.io/en/latest/tune/index.html
from functools import partial
from typing import Callable, Optional, Union
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.nn import Module
from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import SearchAlgorithm
import numpy as np


__all__ = [
    'SimpleRayTuner'
]


class SimpleRayTuner:
    """A simple implementation of the Ray Tuner interface.

    Notes
    -----
    The objective function has to include the **whole** training procedure (incl. graphs loading, model application
    and validation). This means it either needs to be able to take training graphs, validation graphs and a model,
    or it needs to load these inside the function, as each run is separately instantiated models. See the
    ``adam_objective`` function for information on how to use this interface.

    Notes
    ------
    This function does **not work** with BayesOptSearch at the moment.

    Parameters
    -----------
    objective : Callable
        The objective function to be given the ray tune search space optimizer. Could be something like
        ``adam_objective`` in optim.objectives.
    search_space : dict
        A dictionary with the search spaces of each hyperparameter with their name as a key (e.g. ``{'lr':
        ray.tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1, 1e-2])}``).
    train_loader: Data or DataLoader
        The training graphs loader to be used for training **if** one of the native objectives like
        ``adam_objective`` in optim.objectives is used. Default: None
    train_loader: Data or DataLoader
        The validation graphs loader to be used for training **if** one of the native objectives like
        ``adam_objective`` in optim.objectives is used. Default: None
    model: Module
        The model to be used for training and optimization **if** one of the native objectives like
        ``adam_objective`` in optim.objectives is used. Default: None
    metric: str
        The metric to be used in the Tuner. It is specified inside the objective function under
        ``train.report(metric: metric_value)``. Default: 'mean_squared_error' (used in native objective functions)
    mode: str
        Specifies the optimization mode like ``max`` or ``min``. Default: ``min``
    search_algo: SearchAlgorithm
        Gives the Ray Tuner its optimization algorithm like ``HyperOptSearch``. Default: ``HyperOptSearch``
    train_iterations: int
        The number of training iterations for each optimization run. Default: 15

    """

    def __init__(self, objective: Callable, search_space: dict,
                 train_loader: Union[None, Data, DataLoader], val_loader: Union[None, Data, DataLoader],
                 model: Union[None, Module],
                 metric: str = 'mean_squared_error',
                 mode: str = 'min', search_algo: SearchAlgorithm = HyperOptSearch,
                 train_iterations: int = 15):

        # 'objective' should take a train_loader, val_loader and model as input or be configured as such
        if objective.__module__ == 'grape_chem.optim.objectives':
            objective = partial(objective, train_loader=train_loader, val_loader=val_loader, model=model)

        if search_algo.__module__ == 'ray.tune.search.bayesopt.bayesopt_search':
            np.float = float
            search_algo = partial(search_algo, metric = metric,mode=mode)

        self.tuner = tune.Tuner(
            objective,
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                search_alg=search_algo(),
            ),
            run_config=train.RunConfig(
                stop={"training_iteration": train_iterations},
            ),
            param_space=search_space,
        )

    def fit(self):
        self.results = self.tuner.fit()
        print('Best config is: ', self.results.get_best_result().config)

    def __repr__(self):
        print(self.tuner)
