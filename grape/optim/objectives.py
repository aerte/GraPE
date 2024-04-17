from typing import Callable, Union

import torch
from torch.nn.functional import mse_loss
from torch.nn import Module
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from ray import train, tune

from grape.utils.model_utils import train_epoch, val_epoch

__all__ = [
    'adam_default_search_space',
    'adam_objective'
]

def adam_default_search_space():
    """Return a dictionary of default search space for Adam.

    Returns
    -------
    dict

    """
    return {'lr': tune.uniform(1e-4, 1e-2), 'weight_decay': tune.uniform(1e-4, 1e-2),}

def adam_objective(config, train_loader: Union[Data, DataLoader] ,
                   val_loader: Union[Data, DataLoader],
                   model: Module, loss_func: Callable = mse_loss,
                   batch_size: int = 32):
    """An objective function to optimize the Adam optimizer hyperparameters.

    Parameters
    ----------
    config: dict
        Keyword for the ray.Tuner class. It Is a dictionary of the hyperparameters with 'lr' and 'weight_decay'.
    train_loader: Union[Data, DataLoader]
        The training data, either in the Data or DataLoader format.
    val_loader: Union[Data, DataLoader]
        The validation data, either in the Data or DataLoader format.
    model: Module
        The model to be optimized. It has to be a subclass of torch.nn.Module and should be initialized beforehand.
    loss_func: Callable
        Any loss function that will be used to train the model. Default: mse_loss
    batch_size: int
        The batch size to be used for training the model. Default: 32

    """

    model = model()

    if loss_func is None:
        loss_func = mse_loss

    if not isinstance(train_loader, DataLoader):
        train_loader = DataLoader(train_loader, batch_size = batch_size)

    if not isinstance(val_loader, DataLoader):
        val_loader = DataLoader(val_loader, batch_size = batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    while True:
        train_epoch(model=model, train_loader=train_loader, optimizer=optimizer, loss_func=loss_func)
        loss = val_epoch(model=model, val_loader=val_loader, loss_func=loss_func)
        train.report({'mean_squared_error': loss}) # Reporting the mean accuracy to RayTune