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
from grape_chem.utils import DataSet
import os
import dgl
import mlflow
import mlflow.pytorch

__all__ = [
    'EarlyStopping',
    'reset_weights',
    'load_model',
    'load_model_from_data',
    'train_model',
    'train_epoch',
    'val_epoch',
    'test_model',
    'pred_metric',
    'return_hidden_layers',
    'set_seed',
    'rescale_arrays'
]

# import grape_chem.models


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
                self.save_checkpoint(model=model)
            self.stop = True

    def reset(self):
        self.best_score = np.inf
        self.counter = 0
        self.stop = False

    def save_checkpoint(self, model: Module):
        print("Saving model with checkpoint: ", self.model_name)
        if model.dataset_dict is not None:
            checkpoint = create_checkpoint(model)
            torch.save(checkpoint, self.model_name)
        else:
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


def load_model(model_class, model_path, device=None):
    """
    Load a model from a .pt file.

    Args:
        model_class (string): The class of the model to instantiate. Can choose from DMPNN and AFP
        model_path (str): Path to the .pt file containing the model state_dict.
        device (torch.device, optional): The device to load the model onto. Defaults to CUDA if available, otherwise CPU.

    Returns:
        torch.nn.Module: The loaded model.
    """
    allowed_models = ['dmpnn', 'afp']
    print(f"Loading model from {model_path}")
    if model_class.lower() not in model_path.lower():
        print((f"Model class {model_class} does not match with the model saved in {model_path}. Make sure they match."))
        return -1, -1
    if model_class.lower() not in allowed_models:
        raise ValueError(f"Invalid model class {model_class}. Must be one of {allowed_models}")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' not in checkpoint.keys():
        raise ValueError("Model state_dict not found in the checkpoint.")
    if 'model_params' not in checkpoint.keys():
        raise ValueError("Model parameters not found in the checkpoint.")
    state_dict = checkpoint['model_state_dict']
    model_params = checkpoint['model_params']
    #print("checkpoint state_dict: ", state_dict)
    print("checkpoint model_params: ", model_params)
    try:
        # Instantiate the model
        print("Model class: ", model_class)
        if model_class.lower() == 'afp':
            from grape_chem.models import AFP
            model = AFP(node_in_dim=model_params["node_in_dim"], edge_in_dim=model_params["edge_in_dim"], hidden_dim=model_params["hidden_dim"], out_dim=model_params["out_dim"], num_layers_atom=model_params["num_layers_atom"], num_layers_mol=model_params["num_layers_mol"], dropout=model_params["dropout"], mlp_out_hidden=model_params["mlp_out_hidden"], num_global_feats=model_params["num_global_feats"])
        if model_class.lower() == 'dmpnn':
            from grape_chem.models import DMPNN
            print("hi")
            model = DMPNN(node_in_dim=model_params["node_dim"], edge_in_dim=model_params["edge_dim"], node_hidden_dim=model_params["hidden_size"], mlp_out_hidden=model_params["mlp_out_hidden"], num_global_feats=model_params["num_global_feats"])
            print("__Something")
        # Load the state dictionary
        dataset = DataSet(allowed_atoms=model_params["allowed_atoms"], atom_feature_list=model_params["atom_feature_list"], bond_feature_list=model_params["bond_feature_list"], mean=model_params["data_mean"], std=model_params["data_std"])
        # Load the state dict into the model
        
        model.load_state_dict(state_dict)
        # Set the model to evaluation mode
        model.eval()
        # Move the model to the appropriate device
        model.to(device)
        
        return model, dataset
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None
    
def load_model_from_data(data, config, device):
    """
    Load and initialize the model based on the provided data and configuration.

    Parameters
    ----------
    data : Data
        A data object containing the input features and relevant properties for the model.
    config : dict
        A configuration dictionary containing model specifications such as hidden dimensions, model name, and layer information.
    device : str
        The device on which the model should be loaded (e.g., 'cpu' or 'cuda').
    Returns
    -------
    model : torch.nn.Module
        The initialized model based on the specified configuration.
    """

    # Extract node and edge input dimensions from the sample data
    sample = data[50]  # Get a sample from the dataset
    node_in_dim = sample.x.shape[1]  # Number of input features for nodes
    edge_in_dim = sample.edge_attr.shape[1]  # Number of input features for edges


    
    # Get MLP hidden layers from configuration
    mlp = return_hidden_layers(config['mlp_layers'])
    
    # Prepare a dataset dictionary for additional model context
    dataset_dict = {
        'allowed_atoms': data.allowed_atoms, 
        'atom_feature_list': data.atom_feature_list, 
        'bond_feature_list': data.bond_feature_list, 
        'data_mean': data.mean, 
        'data_std': data.std, 
        'data_name': data.data_name
    }
    
    # Initialize model based on the specified model name
    if 'dmpnn' in config['model_name'].lower():
        from grape_chem.models import DMPNN
        model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, 
                      node_hidden_dim=config['hidden_dim'], dropout=config['dropout'], 
                      mlp_out_hidden=mlp, dataset_dict=dataset_dict)
    elif 'afp' in config['model_name'].lower():
        from grape_chem.models import AFP
        model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, 
                    out_dim=1, dataset_dict=dataset_dict)
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")
    
    # Log model parameters to MLflow if mlflow is active
    if mlflow.active_run():
        mlflow.log_params({
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "out_dim": 1
        })
        
        # Count and log the number of learnable parameters in the model
        num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("num_learnable_params", num_learnable_params)
    
    # Move model to the specified device (CPU or GPU)
    model = model.to(device)

    return model

##########################################################################
########### Training and Testing functions ###############################
##########################################################################


def train_model(model: torch.nn.Module, loss_func: Union[Callable,str], optimizer: torch.optim.Optimizer,
                train_data_loader: Union[list, Data, DataLoader], val_data_loader: Union[list, Data, DataLoader],
                device: str = None, epochs: int = 50, batch_size: int = 32,
                early_stopper: EarlyStopping = None, scheduler: lr_scheduler = None,
                tuning: bool = False, model_name:str = None) -> tuple[list,list]:
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
        A list of Data objects or the DataLoader directly to be used as the training graphs.
    val_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the validation graphs.
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
    tuning: bool
        Will turn off the early stopping, meant for hyperparameter optimziation.
    model_name:str
        If given, it will be used to save it if early stopping did not set in. Default: None

    Returns
    ---------
    tuple
        Training and test loss arrays.

    """
    loss_functions = {
        'mse': torch.nn.functional.mse_loss,
        'mae': torch.nn.functional.l1_loss
    }

    if batch_size is None:
        batch_size = len(train_data_loader)

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

            if mlflow.active_run():
                mlflow.log_metric("train_loss_inside_train_model", loss_train, step=i)
                mlflow.log_metric("val_loss_inside_train_model", loss_val, step=i)

            if i%2 == 0:
                pbar.set_description(f"epoch={i}, training loss= {loss_train:.3f}, validation loss= {loss_val:.3f}")

            if scheduler is not None:
                scheduler.step(loss_val)


            if early_stopper is not None:
                #print(f'early stopper: {early_stopper}')
                early_stopper(val_loss=loss_val, model=model)
                if early_stopper.stop:
                    early_stopper.stop_epoch = i-early_stopper.patience
                    if tuning:
                        pass
                    else:
                        break

            pbar.update(1)
        #print(f'eary stopper: {early_stopper}')
#DataSet(smiles=smiles, target=target, global_features=global_features, allowed_atoms=allowed_atoms, atom_feature_list=atom_feature_list, bond_feature_list=bond_feature_list, log=False, only_organic=False, filter=True, allow_dupes=True)
    
        if early_stopper is not None:
            print("Early stopper: ", early_stopper)
            if early_stopper.stop is False and model_name is not None:
                print("Early stopping did not set in, saving model.")
                early_stopper.save_checkpoint(model, model_name)
                print(f'Model saved at: {model_name}')
    
        else:
            if model_name is not None:
                print("Early stopping did not set in, saving model.")
                early_stopper.save_checkpoint(model, model_name)
                print(f'Model saved at: {model_name}')

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

    avg_loss = loss/it
    return avg_loss


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

    avg_loss = loss / it

    return avg_loss


##############################################################################################################
################################# Model testing ##############################################################s
##############################################################################################################



def test_model(model: torch.nn.Module, test_data_loader: Union[list, Data, DataLoader],
                device: str = None, batch_size: int = 32, return_latents: bool = False) -> (
        Union[Tensor, tuple[Tensor,Tensor], tuple[Tensor, Tensor, list]]):
    """Auxiliary function to test a trained model and return the predictions as well as the latent node
    representations. If a loss function is specified, then it will also return a list containing the testing losses.
    Can initialize DataLoaders if only list of Data objects are given.

        Notes
    ------
    This function is made for native models. If the model does not include a return_latents parameter, then consider
    building a custom test function.

    Parameters
    ------------
    model: torch.nn.Module
        Model that will be trained and tested. Has to be a torch Module.
    test_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the test graphs.
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
        Predictions of the model. If return_latents is True, then it will return a tuple of predictions and latents.

    """


    device = torch.device('cpu') if device is None else device

    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader(test_data_loader, batch_size = batch_size)

    model.eval()
    all_preds = []
    all_targets = []

    with tqdm(total = len(test_data_loader)) as pbar:

        for idx, batch in enumerate(test_data_loader):
            # TODO: Broaden use of return_latents
            out = model(batch.to(device))
            all_preds.append(out)
            all_targets.append(batch.y)
            if return_latents:
               lat = model(batch.to(device), return_lats=True)

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

    log_preds = torch.cat(all_preds).detach()
    log_targets = torch.cat(all_targets).detach()

    if mlflow.active_run():
        metrics = pred_metric(log_preds, log_targets, metrics='all', print_out=False)
        print("############## METRICS ##########################################",metrics, type(metrics))
        # Already logging them in pred_metric
        # metrics =  {'mse': 0.21034354, 'rmse': 0.45863226, 'sse': 10.517177, 'mae': 0.36576897, 'r2': 0.7754248380661011, 'mre': 1.2537103, 'mdape': 46.80940508842468}
        #mlflow.log_metrics(metrics)
        
    if return_latents:
        return preds, latents
    return preds


##########################################################################
########### Prediction Metrics ###########################################
##########################################################################

def rescale_arrays(arrays: Union[Tensor, tuple[Tensor,Tensor], list], data:object = None,
                   mean:float = None, std: float = None) -> Union[Tensor, tuple[Tensor,Tensor], list]:
    """ A helper function to rescale the input arrayso or tensors based on a GraPE dataset or a given mean/std.

    Parameters
    ----------
    arrays: Union[Tensor, tuple[Tensor,Tensor], list]
        The input arrays or tensors to be rescaled. Can be singular, a tuple or a list of tensors.
    data: object
        A GraPE DataSet or GraphDataSet object. Default: None
    mean: float
        The mean value of the scaling. Default: None
    std: float
        The std value of the scaling. Default: None


    Returns
    -------
    All arrays given rescaled.


    """

    assert data is not None or (mean is not None and std is not None), ('Either a graphs object from GraPE'
            'with a valid mean and std, or a mean and std must be given.')

    out = []
    for array in arrays:
        if isinstance(array, Tensor):
            array = array.cpu().detach().numpy()
        if data is not None:
            out.append(data.rescale(array, data.mean, data.std))
        elif mean is not None and std is not None:
            out.append((array * std)+mean)

    return out


def pred_metric(prediction: Union[Tensor, ndarray], target: Union[Tensor, ndarray],
                metrics: Union[str,list[str]] = 'mse', print_out: \
                bool = True, rescale_data: DataSet = None) -> list[float]:
    """A function to evaluate continuous predictions compared to targets with different metrics. It can
    take either Tensors or ndarrays and will automatically convert them to the correct format. Partly makes use of
    sklearn and their implementations. The options for metrics are:

    * ``MSE``: Mean Squared Error
    * ``RMSE``: Root Mean Squared Error
    * ``SSE``: Sum of Squared Errors
    * ``MAE``: Mean Average Error
    * ``R2``: R-squared Error
    * ``MRE``: Mean Relative Error, which is implemented as:
    * ``MDAPE``: Median Absolute Percentage Error

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
        [``mse``, ``rmse``, ``sse``, ``mae``, ``r2``, ``mre``, ``mdape``] or 'all' for every option. Default: 'mse'
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
        metrics = ['mse','rmse','sse','mae','r2','mre', 'mdape']

    results = dict()
    prints = []
    delta = 1e-12
    target = target+delta

    for metric_ in metrics:
        if metric_ == 'mse':
            results['mse'] = mean_squared_error(target, prediction)
            prints.append(f'MSE: {results["mse"]:.3f}')
        elif metric_ == 'rmse':
            results['rmse'] = np.sqrt(mean_squared_error(target, prediction))
            prints.append(f'RMSE: {results["rmse"]:.3f}')
        elif metric_ == 'mae':
            results['mae'] = mean_absolute_error(target, prediction)
            prints.append(f'MAE: {results["mae"]:.3f}')
        elif metric_ == 'r2':
            results['r2'] = r2_score(target, prediction)
            prints.append(f'R2: {results["r2"]:.3f}')
        elif metric_ == 'sse':
            results['sse'] = np.sum(np.square(target-prediction))
            prints.append(f'SSE: {results["sse"]:.3f}')
        elif metric_ == 'mre':
            results['mre'] = np.mean(np.abs((target - prediction) / target))
            prints.append(f'MRE: {results["mre"]:.3f}')
        elif metric_ == 'mdape':
            results['mdape'] = np.median(np.abs((target - prediction) / target)) * 100
            prints.append(f'MDAPE: {results["mdape"]:.3f}')

    # Log all relevant metrics to MLFlow if running
    if mlflow.active_run():
        for key, value in results.items():
            mlflow.log_metric(key, value)

    if print_out:
        for out in prints:
            print(out)

    return results


#######################################################################################################
#################################### General tools ####################################################
#######################################################################################################

def create_checkpoint(model: Module) -> dict:
    """Creates a dictionarty that contains the model state_dict and model parameters.

    Args:
        model (torch.nn.Module): The model to save.

    Returns:
        dict: A dictionary containing the model state_dict and model parameters.
    """
    possible_attributes = [
        "node_in_dim", "edge_in_dim", "node_dim", "edge_dim","hidden_dim", "hidden_size", "out_dim",
        "num_layers_atom", "num_layers_mol", "dropout", "regressor",
        "mlp_out_hidden", "rep_dropout", "num_global_feats",
        "allowed_atoms", "atom_feature_list", "bond_feature_list",
        "data_mean", "data_std"
    ]

    # Create model_params dictionary dynamically
    model_params = {attr: getattr(model, attr) for attr in possible_attributes if hasattr(model, attr)}
    return {
        'model_state_dict': model.state_dict(),
        'model_params': model_params
    }



def set_seed(seed:int = 42, numpy_off: bool = False, is_ensemble: bool = False):
    """ A function to set the random seed for reproducibility. This includes NumPy, PyTorch and DGL. The
    primary purpose of this function is to guarantee that the same weights are used for model initialization.
    This is important when optimizing and training on the optimized values.

    Parameters
    ----------
    seed: int
        The seed that will be used for *all* random number generators specified in the description. Default: 42
    numpy_off: bool
        If true, will not seed the numpy random number generator. Default: False
    is_ensemble: bool
        If true, will not seed the torch random number generator. Usually turned off for
         training ensemble models. Default: False

    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    # turn it off during optimization
    if not numpy_off:
        np.random.seed(seed)
    if not is_ensemble:
        torch.manual_seed(seed)  # annotate this line when ensembling
    dgl.random.seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # Faster than the command below and may not fix results
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

def return_hidden_layers(num):
    """ Returns a list of hidden layers, starting from 2**num*32, reducing the hidden dim by 2 every step.

    Example
    --------

    >>>return_hidden_layers(3)

    [256, 128, 64]
    """
    return [2**i*32 for i in range(num, 0,-1)]







