from typing import Callable, Union
import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.optim import lr_scheduler
from numpy import ndarray
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from grape.utils import DataSet

__all__ = [
    'EarlyStopping',
    'reset_weights',
    'train_model',
    'train_epoch',
    'val_epoch',
    'test_model',
    'pred_metric'
]

# import grape.models


##########################################################################
########### Model utils ##################################################
##########################################################################


class EarlyStopping:
    """Simple early stopper for any PyTorch Model.

    Serves as a simple alternative to the sophisticated implementations from
    PyTorch-Lightning (https://pytorch-lightning.readthedocs.io/en/stable/) or similar.

    Parameters
    -----------
    patience : int, optional
        Patience for early stopping.
        It will stop training if the validation loss doesn't improve after a
        given number of 'patient' epochs. Default: 15.
    min_delta : float, optional
        The minimum loss improvement needed to qualify as a better model. Default: 1e-3
    model_name: str
        A name for the model, will be used to save it. Default: 'best_model'
    skip_save: bool
        Whether to skip saving the model. Default: False

    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-3, model_name = 'best_model', skip_save: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.model_name = model_name
        self.best_score = np.inf
        self.counter = 0
        self.stop = False
        self.model_name = model_name + '.pt'
        self.stop_epoch = 0
        self.skip_save = skip_save

    def __call__(self, val_loss: Tensor, model: Module):
        if val_loss < self.best_score + self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            if not self.skip_save:
                self.save_checkpoint(model=model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f'Early stopping reached with best validation loss {self.best_score:.4f}')
            if not self.skip_save:
                print(f'Model saved at: {self.model_name}')
            self.stop = True



    def save_checkpoint(self, model: Module):
        torch.save(model.state_dict(), self.model_name)


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


def train_model(model: torch.nn.Module, loss_func: Union[Callable,str], optimizer: torch.optim.Optimizer,
                train_data_loader: Union[list, Data, DataLoader], val_data_loader: Union[list, Data, DataLoader],
                device: str = None, epochs: int = 50, batch_size: int = 32,
                early_stopper: EarlyStopping = None, scheduler: lr_scheduler = None) -> tuple[list,list]:
    """Auxiliary function to train and test a given model and return the (training, test) losses.
    Can initialize DataLoaders if only lists of Data objects are given.

    Parameters
    -------------
    model: torch.nn.Module
        Model that will be trained and tested. Has to be a torch Module.
    loss_func: Callable or str
        Loss function like F.mse_loss that will be used as a loss. Or a string that can be one of
        [``mse``,``mae``]
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
    early_stopper: EarlyStopping
        Optional EarlyStopping class that will apply the defined early stopping method. Default: None
    scheduler: lr_scheduler
        Optional learning rate scheduler that will take a step after validation. Default: None

    Returns
    ---------
    tuple
        Training and test loss arrays.

    """
    loss_functions = {
        'mse': torch.nn.functional.mse_loss,
        'mae': torch.nn.functional.l1_loss
    }

    if isinstance(loss_func, str):
        loss_func = loss_functions[loss_func]


    device = torch.device('cpu') if device is None else device

    if not isinstance(train_data_loader, DataLoader):
        train_data_loader = DataLoader(train_data_loader, batch_size = batch_size)

    if not isinstance(val_data_loader, DataLoader):
        val_data_loader = DataLoader(val_data_loader, batch_size = batch_size)

    model.train()
    train_loss = []
    val_loss = []

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

            if i%2 == 0:
                    pbar.set_description(f"epoch={i}, training loss= {loss_train:.3f}, validation loss= {loss_val:.3f}")

            if scheduler is not None:
                scheduler.step(loss_val)


            if early_stopper is not None:
                early_stopper(val_loss=loss_val, model=model)
                if early_stopper.stop:
                    early_stopper.stop_epoch = i-early_stopper.patience
                    break

            pbar.update(1)

    return train_loss, val_loss


##############################################################################################################
################ Epoch level train and val ###################################################################
##############################################################################################################

def train_epoch(model: torch.nn.Module, loss_func: Callable, optimizer: torch.optim.Optimizer,
                train_loader, device: str = None):

    if device is None:
        device = torch.device('cpu')

    model.train()

    loss = 0.
    it = 0.

    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(batch.to(device))
        loss_train = loss_func(batch.y, out)

        loss += loss_train.detach().cpu().numpy()
        it += 1.

        loss_train.backward()
        optimizer.step()

    return loss/it


def val_epoch(model: torch.nn.Module, loss_func: Callable, val_loader, device: str = None):
    if device is None:
        device = torch.device('cpu')
    model.eval()

    loss = 0.
    it = 0.

    for idx, batch in enumerate(val_loader):
        out = model(batch.to(device))
        loss_val = loss_func(batch.y, out)
        loss += loss_val.detach().cpu().numpy()
        it += 1.

    return loss/it


##############################################################################################################
################################# Model testing ##############################################################s
##############################################################################################################



def test_model(model: torch.nn.Module, loss_func: Union[Callable,str,None], test_data_loader: Union[list, Data, DataLoader],
                device: str = None, batch_size: int = 32, return_latents: bool = False) -> (
        Union[Tensor, tuple[Tensor,Tensor], tuple[Tensor, Tensor, list]]):
    # TODO: Change this function to something intuitive
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

    loss_functions = {
        'mse': torch.nn.functional.mse_loss,
        'mae': torch.nn.functional.l1_loss
    }

    #if not isinstance(model, grape.models.SimpleGNN):
    #    return_latents = False



    if isinstance(loss_func, str):
        loss_func = loss_functions[loss_func]

    device = torch.device('cpu') if device is None else device

    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader(test_data_loader, batch_size = batch_size)

    model.eval()

    with tqdm(total = len(test_data_loader)) as pbar:

        temp = np.zeros(len(test_data_loader))

        for idx, batch in enumerate(test_data_loader):
            # TODO: Broaden use of return_latents
            #if return_latents and isinstance(model, grape.models.SimpleGNN):
            #    out, lat = model(batch.to(device), return_latents=True)
            #else:
            out = model(batch.to(device))

            if loss_func is not None:
                temp[idx] = loss_func(batch.y, out).detach().cpu().numpy()

            # Concatenate predictions and latents
            if idx == 0:
                preds = out
                #if return_latents:
                #    latents = lat
            else:
                preds = torch.concat([preds,out],dim=0)
                #if return_latents:
                #    latents = torch.concat((latents, lat), dim=0)

            pbar.update(1)

    loss_test = np.mean(temp)

    # TODO: fix this mess

    #if loss_func is not None and return_latents:
    #    print(f'Test loss: {loss_test:.3f}')
    #    return preds, latents, loss_test
    #elif loss_func is not None:
    #    print(f'Test loss: {loss_test:.3f}')
    #    return preds, loss_test
    #elif return_latents:
    #    return preds, latents
    return preds


##########################################################################
########### Prediction Metrics ###########################################
##########################################################################


def pred_metric(prediction: Union[Tensor, ndarray], target: Union[Tensor, ndarray],
                metrics: Union[str,list[str]] = 'mse', print_out: \
                bool = True, rescale_data: DataSet = None) -> list[float]:
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
    if rescale_data is not None:
        prediction =  rescale_data.rescale_data(prediction)
        target = rescale_data.rescale_data(target)

    if metrics == 'all':
        metrics = ['mse','rmse','sse','mae','r2','mre']

    results = dict()
    prints = []

    for metric_ in metrics:
        if metric_ == 'mse':
            results['mse'] = mean_squared_error(target, prediction)
            prints.append(f'MSE: {mean_squared_error(target, prediction):.3f}')
        elif metric_ == 'rmse':
            results['rmse'] = np.sqrt(mean_squared_error(target, prediction))
            prints.append(f'RMSE: {np.sqrt(mean_squared_error(target, prediction)):.3f}')
        elif metric_ ==  'sse':
            results['sse'] = np.sum((target-prediction)**2)
            prints.append(f'SSE: {np.sum((target-prediction)**2):.3f}')
        elif metric_ ==  'mae':
            results['mae'] = mean_absolute_error(target, prediction)
            prints.append(f'MAE: {mean_absolute_error(target, prediction):.3f}')
        elif metric_ ==  'r2':
            results['r2'] = r2_score(target, prediction)
            prints.append(f'R2: {r2_score(target, prediction):.3f}')
        elif metric_ ==  'mre':
            results['mre'] = np.mean(np.abs((target-prediction))/target)*100
            prints.append(f'MRE: {np.mean(np.abs(target - prediction) / target) * 100:.3f}%')
            if results['mre'] > 100:
                prints.append(f'Mean relative error is large, here is the median relative error'
                                f':{np.median(np.abs(target-prediction)/target)*100:.3f}%')

    if print_out:
        for out in prints:
            print(out)

    return results








