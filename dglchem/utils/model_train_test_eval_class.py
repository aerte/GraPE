from typing import Callable
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

__all__ = [
    'train_model'
]

# Should be used after the model is initialized
def train_model(model: torch.nn.Module, loss_func: Callable, optimizer: torch.optim.Optimizer,
                train_data_loader: list or DataLoader, test_data_loader: list or DataLoader,
                device: str = None, epochs: int = 50, batch_size: int = 32,
                early_stopping: bool = True, patience: int = 3) -> tuple:
    """Function to train and test a given model. Can initialize DataLoaders if only list of Data objects are given.

    Parameters
    ----------
    model: torch.nn.Module
        Model that will be trained and tested. Has to be a torch Module.
    loss_func: Callable
        Loss function like F.mse_loss that will be used as a loss.
    optimizer: torch.optim.Optimizer
        Torch optimization algorithm like Adam or SDG.
    train_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the training data.
    test_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the test data.
    device: str
        Torch device to be used ('cpu','cuda' or 'mps'). Default: 'cpu'
    epochs: int
        Training epochs. Default: 50
    batch_size: int
        Batch size of the DataLoader if not given directly. Default: 32
    early_stopping: bool
        Decides if early stopping should be used. Default: True
    patience: int
        Decides how many 'bad' epochs can pass before early stopping takes effect. Default: 3

    Returns
    -------
    tuple
        Training and test loss arrays.

    """

    device = torch.device('cpu') if device is None else device

    if not isinstance(train_data_loader, DataLoader):
        train_data_loader = DataLoader(train_data_loader, batch_size = batch_size)

    if not isinstance(train_data_loader, DataLoader) and test_data_loader:
        train_data_loader = DataLoader(train_data_loader, batch_size = batch_size)

    model.train()
    train_loss = []
    test_loss = []

    loss_cut = float('inf')
    counter = 0

    with tqdm(total = epochs) as pbar:
        for i in range(epochs):
            temp = np.zeros(len(train_data_loader))
            for idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                out = model(batch.to(device))
                loss_val = loss_func(batch.y, out)

                temp[idx] = loss_val.detach().cpu().numpy()


                loss_val.backward()
                optimizer.step()

            loss_train = np.mean(temp)
            train_loss.append(loss_train)

            temp = np.zeros(len(test_data_loader))
            for idx, batch in enumerate(test_data_loader):
                out = model(batch.to(device))
                temp[idx] = loss_func(batch.y, out).detach().cpu().numpy()

            loss_test = np.mean(temp)
            test_loss.append(loss_test)

            if i%5 == 0:
                pbar.set_description(f"epoch={i}, training loss= {loss_train:.3f}, validation loss= {loss_test:.3f}")

            if loss_test < loss_cut:
                loss_cut = loss_test
                counter = 0
            else:
                counter+=1

            if counter==patience and early_stopping:
                print('Model hit early stop threshold. Ending training.')
                break


            pbar.update(1)

    return train_loss, test_loss




