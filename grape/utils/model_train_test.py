from typing import Callable
import torch
from torch import Tensor
from numpy import ndarray
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

__all__ = [
    'train_model',
    'test_model',
    'pred_metric'
]


def train_model(model: torch.nn.Module, loss_func: Callable, optimizer: torch.optim.Optimizer,
                train_data_loader: list or DataLoader, val_data_loader: list or DataLoader,
                device: str = None, epochs: int = 50, batch_size: int = 32,
                early_stopping: bool = True, patience: int = 3) -> tuple[list,list]:
    """Auxiliary function to train and test a given model and return the (training, test) losses.
    Can initialize DataLoaders if only list of Data objects are given.

    Parameters
    -------------
    model: torch.nn.Module
        Model that will be trained and tested. Has to be a torch Module.
    loss_func: Callable
        Loss function like F.mse_loss that will be used as a loss.
    optimizer: torch.optim.Optimizer
        Torch optimization algorithm like Adam or SDG.
    train_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the training data.
    val_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the validation data.
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
    ---------
    tuple
        Training and test loss arrays.

    """

    device = torch.device('cpu') if device is None else device

    if not isinstance(train_data_loader, DataLoader):
        train_data_loader = DataLoader(train_data_loader, batch_size = batch_size)

    if not isinstance(val_data_loader, DataLoader):
        val_data_loader = DataLoader(val_data_loader, batch_size = batch_size)

    model.train()
    train_loss = []
    val_loss = []

    loss_cut = float('inf')
    counter = 0

    with tqdm(total = epochs) as pbar:
        for i in range(epochs):
            temp = np.zeros(len(train_data_loader))
            for idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                out = model(batch.to(device))
                loss_train = loss_func(batch.y, out)

                temp[idx] = loss_train.detach().cpu().numpy()


                loss_train.backward()
                optimizer.step()

            loss_train = np.mean(temp)
            train_loss.append(loss_train)

            temp = np.zeros(len(val_data_loader))
            for idx, batch in enumerate(val_data_loader):
                out = model(batch.to(device))
                temp[idx] = loss_func(batch.y, out).detach().cpu().numpy()

            loss_val = np.mean(temp)
            val_loss.append(loss_val)

            if i%5 == 0:
                pbar.set_description(f"epoch={i}, training loss= {loss_train:.3f}, validation loss= {loss_val:.3f}")

            if loss_val < loss_cut:
                loss_cut = loss_val
                counter = 0
            else:
                counter+=1

            if counter==patience and early_stopping:
                print('Model hit early stop threshold. Ending training.')
                break


            pbar.update(1)

    return train_loss, val_loss


def test_model(model: torch.nn.Module, loss_func: Callable, test_data_loader: list or DataLoader,
                device: str = None, batch_size: int = 32) -> tuple:
    """Auxiliary function to validate a trained model and return the validation losses.
    Can initialize DataLoaders if only list of Data objects are given.

    Parameters
    ------------
    model: torch.nn.Module
        Model that will be trained and tested. Has to be a torch Module.
    loss_func: Callable
        Loss function like F.mse_loss that will be used as a loss.
    test_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the test data.
    device: str
        Torch device to be used ('cpu','cuda' or 'mps'). Default: 'cpu'
    batch_size: int
        Batch size of the DataLoader if not given directly. Default: 32

    Returns
    ---------
    float
        Mean test loss over the test set batches.

    """

    device = torch.device('cpu') if device is None else device

    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader(test_data_loader, batch_size = batch_size)

    model.eval()

    with tqdm(total = len(test_data_loader)) as pbar:

        temp = np.zeros(len(test_data_loader))

        for idx, batch in enumerate(test_data_loader):
            out = model(batch.to(device))
            temp[idx] = loss_func(batch.y, out).detach().cpu().numpy()

            pbar.update(1)

    loss_test = np.mean(temp)

    print(f'Test loss: {loss_test:.3f}')

    return loss_test

def pred_metric(prediction: Tensor or ndarray, target: Tensor or ndarray, metrics: str or list[str] = 'mse') -> list[float]:
    """A function to evaluate continuous predictions compared to targets with different metrics. It can
    take either Tensors or ndarrays and will automatically convert them to the correct format. Partly makes use of
    sklearn and their implementations. The options for metrics are:

    * ``MSE``: Mean Squared Error
    * ``SSE``: Sum of Squared Errors
    * ``MAE``: Mean Average Error
    * ``R2``: R-squared Error
    * ``MRE``: Mean Relative Error, which is implemented as: :math:`\frac{1}{N}\sum_`
    .. math:
        \frac{1}{N}\sum\limits_{i=1}^{N}\frac{y_{i}-f(x_{i})}{y_{i}}\cdot100

    **If a list of metrics is given, then a list of equal length is returned.**

    Parameters:
    -----------
    prediction: Tensor or ndarray
        A prediction array or tensor generated by some sort of model.
    target: Tensor or ndarray
        The target array or tensor corresponding to the prediction.
    metrics: str or list[str]
        A string or a list of strings specifying what metrics should be returned. The options are:
        [``mse``, ``sse``, ``mae``, ``r2``, ``mre``]. Default: 'mse'

    Returns:
    --------
    list[float]
        A list equal to the number of metrics specified contained the corresponding results.

    """

    if isinstance(prediction, Tensor):
        prediction = prediction.cpu().detach().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().detach().numpy()

    results = []

    for metric_ in metrics:
        match metric_:
            case 'mse':
                results.append(mean_squared_error(target, prediction))
            case 'sse':
                results.append(np.sum((target-prediction)**2))
            case 'mae':
                results.append(mean_absolute_error(target, prediction))
            case 'r2':
                results.append(r2_score(target, prediction))
            case 'mre':
                results.append(np.sum((target-prediction)/target)*100)

    return results









