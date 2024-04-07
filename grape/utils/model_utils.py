from typing import Callable, Union
import torch
from torch import Tensor
from torch.nn import Module, Sequential
from numpy import ndarray
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

__all__ = [
    'reset_weights',
    'train_model',
    'test_model',
    'pred_metric'
]

##########################################################################
########### Model utils ##################################################
##########################################################################

def reset_weights(model: Module):
    """Taken from https://discuss.pytorch.org/t/reset-model-weights/19180/12. It recursively resets the models
    weights as well any children's weights. Works on models as well.

    Parameters
    ----------
    layer: Module


    """
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()
    else:
        if hasattr(model, 'children'):
            for child in model.children():
                reset_weights(child)



##########################################################################
########### Training and Testing functions ###############################
##########################################################################


def train_model(model: torch.nn.Module, loss_func: Callable, optimizer: torch.optim.Optimizer,
                train_data_loader: Union[list, Data, DataLoader], val_data_loader: Union[list, Data, DataLoader],
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


def test_model(model: torch.nn.Module, loss_func: Union[Callable,None], test_data_loader: Union[list, Data, DataLoader],
                device: str = None, batch_size: int = 32, return_latents: bool = False) -> (
        Union[Tensor, tuple[Tensor,Tensor], tuple[Tensor, Tensor, list]]):
    """Auxiliary function to test a trained model and return the predictions as well as the latents node
    representations. If a loss function is specified, then it will also return a list containing the testing losses.
    Can initialize DataLoaders if only list of Data objects are given.

    Parameters
    ------------
    model: torch.nn.Module
        Model that will be trained and tested. Has to be a torch Module.
    loss_func: Callable or None
        Loss function like F.mse_loss that will be used as a loss or None, in which case just the predictions are
        returned.
    test_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the test data.
    device: str
        Torch device to be used ('cpu','cuda' or 'mps'). Default: 'cpu'
    batch_size: int
        Batch size of the DataLoader if not given directly. Default: 32
    return_latents: bool
        Decides is the latents should be returned. **If used, the model must include return_latent statement**. Default:
        False

    Returns
    ---------
    float
        Mean test loss over the test set batches.

    Notes
    ------
    This function is made for native models. If the model does not include a return_latents parameter, then consider
    building a custom test function.

    """

    device = torch.device('cpu') if device is None else device

    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader(test_data_loader, batch_size = batch_size)

    model.eval()

    with tqdm(total = len(test_data_loader)) as pbar:

        temp = np.zeros(len(test_data_loader))

        for idx, batch in enumerate(test_data_loader):
            # TODO: Broaden use of return_latents
            if return_latents:
                out, lat = model(batch.to(device), return_latents=True)
            else:
                out = model(batch.to(device), return_latents=False)

            if loss_func is not None:
                temp[idx] = loss_func(batch.y, out).detach().cpu().numpy()

            # Concatenate predictions and latents
            if idx == 0:
                preds = out
                if return_latents:
                    latents = lat
            else:
                preds = torch.concat([preds,out],dim=0)
                if return_latents:
                    latents = torch.concat((latents, lat), dim=0)

            pbar.update(1)

    loss_test = np.mean(temp)

    if loss_func is not None and return_latents:
        print(f'Test loss: {loss_test:.3f}')
        return preds, latents, loss_test
    elif loss_func is not None:
        print(f'Test loss: {loss_test:.3f}')
        return preds, loss_test
    elif return_latents:
        return preds, latents
    return preds


##########################################################################
########### Prediction Metrics ###########################################
##########################################################################


def pred_metric(prediction: Union[Tensor, ndarray], target: Union[Tensor, ndarray],
                metrics: Union[str,list[str]] = 'mse', print_out: \
                bool = True) -> list[float]:
    """A function to evaluate continuous predictions compared to targets with different metrics. It can
    take either Tensors or ndarrays and will automatically convert them to the correct format. Partly makes use of
    sklearn and their implementations. The options for metrics are:

    * ``MSE``: Mean Squared Error
    * ``SSE``: Sum of Squared Errors
    * ``MAE``: Mean Average Error
    * ``R2``: R-squared Error
    * ``MRE``: Mean Relative Error, which is implemented as:

    ..math:
        \frac{1}{N}\sum\limits_{i=1}^{N}\frac{y_{i}-f(x_{i})}{y_{i}}\cdot100

    **If a list of metrics is given, then a list of equal length is returned.**

    Parameters
    -----------
    prediction: Tensor or ndarray
        A prediction array or tensor generated by some sort of model.
    target: Tensor or ndarray
        The target array or tensor corresponding to the prediction.
    metrics: str or list[str]
        A string or a list of strings specifying what metrics should be returned. The options are:
        [``mse``, ``sse``, ``mae``, ``r2``, ``mre``] or 'all' for every option. Default: 'mse'
    print_out: bool
        Will print out formatted results if True. Default: True

    Returns
    --------
    list[float]
        A list equal to the number of metrics specified contained the corresponding results.

    """

    if isinstance(prediction, Tensor):
        prediction = prediction.cpu().detach().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().detach().numpy()
    if not isinstance(metrics, list) and metrics != 'all':
        metrics = [metrics]

    if metrics == 'all':
        metrics = ['mse','sse','mae','r2','mre']

    results = dict()
    prints = []

    for metric_ in metrics:
        match metric_:
            case 'mse':
                results['mse'] = mean_squared_error(target, prediction)
                prints.append(f'MSE: {mean_squared_error(target, prediction):.3f}')
            case 'sse':
                results['sse'] = np.sum((target-prediction)**2)
                prints.append(f'SSE: {np.sum((target-prediction)**2):.3f}')
            case 'mae':
                results['mae'] = mean_absolute_error(target, prediction)
                prints.append(f'MAE: {mean_absolute_error(target, prediction):.3f}')
            case 'r2':
                results['r2'] = r2_score(target, prediction)
                prints.append(f'R2: {r2_score(target, prediction):.3f}')
            case 'mre':
                results['mre'] = np.sum((target-prediction)/target)*100
                prints.append(f'MRE: {np.sum((target-prediction)/target)*100:.3f}%')

    if print_out:
        for out in prints:
            print(out)

    return results









