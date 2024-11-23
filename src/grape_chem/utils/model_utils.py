from typing import Callable, Union, List, Tuple, Optional
import torch
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataloader
from torch.nn import Module, Sequential
from torch.optim import lr_scheduler
from numpy import ndarray
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
from grape_chem.utils import DataSet
import os
import dgl

__all__ = [
    'EarlyStopping',
    'reset_weights',
    'train_model',
    'train_model_jit',
    'train_epoch',
    'val_epoch',
    'test_model',
    'pred_metric',
    'return_hidden_layers',
    'set_seed',
    'rescale_arrays',
    'test_model_jit',
    'train_epoch_jittable',
    'val_epoch_jittable',
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
            #TODO: make into logger instead
            #print(f'Early stopping reached with best validation loss {self.best_score:.4f}')
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


def batch_implicit_with_frag(data_list):
    """
    implicit because we're not actually naming the attributes of the batch
    only works when there IS fragmentation for now, else fails
    """
    # Handle the graphs, fragments, and motifs separately if necessary
    graphs = [item[0] for item in data_list]
    frag_graphs = [item[1] for item in data_list]
    motif_graphs = [item[2] for item in data_list]

    # Use PyTorch Geometric's batching function for each part
    batched_graph = Batch.from_data_list(graphs)
    batched_frag = Batch.from_data_list(frag_graphs) if frag_graphs else None
    batched_motif = Batch.from_data_list(motif_graphs) if motif_graphs else None

    return batched_graph, batched_frag, batched_motif

##########################################################################
########### Training and Testing functions ###############################
##########################################################################


def train_model(model: torch.nn.Module, loss_func: Union[Callable,str], optimizer: torch.optim.Optimizer,
                train_data_loader: Union[list, Data, DataLoader], val_data_loader: Union[list, Data, DataLoader],
                device: str = None, epochs: int = 50, batch_size: int = 32,
                early_stopper: EarlyStopping = None, scheduler: lr_scheduler = None,
                tuning: bool = False, model_name:str = None, model_needs_frag : bool = False,) -> tuple[list,list]:
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

    if isinstance(loss_func, str):
        loss_func = loss_functions[loss_func]


    device = torch.device('cpu') if device is None else device

    exclude_keys = None
    #in some cases this dataloader shouldn't batch the graphs that result from fragmentations with the rest
    if not model_needs_frag:
        if hasattr(train_data_loader, "fragmentation"):
            if train_data_loader.fragmentation is not None:
                exclude_keys = ["frag_graphs", "motif_graphs"]
    

    if not isinstance(train_data_loader, DataLoader):
        train_data_loader = DataLoader(train_data_loader, batch_size = batch_size, exclude_keys=exclude_keys,)

    if not isinstance(val_data_loader, DataLoader):
        val_data_loader = DataLoader(val_data_loader, batch_size = batch_size, exclude_keys=exclude_keys,)

    model.train()
    train_loss = []
    val_loss = []

    def handle_heterogenous_sizes(y, out):
        """
        Unfortunate function due to the fact that we don't
        have a standard return type for all our models.

        Ideally we should have different training loops or 
        standardized model outputs but this will do for now.
        """
        if not isinstance(out, torch.Tensor):
            return out
        if y.dim() == out.dim():
            return out
        return out.squeeze() #needed for some models

    def move_to_device(data, device):
        """
        a wrapper for all the calls to properly move 
        nested and batched PyG Data to a cuda device
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, list):
            return [move_to_device(item, device) for item in data]
        elif isinstance(data, dict):
            return {key: move_to_device(value, device) for key, value in data.items()}
        else:
            return data

    with tqdm(total = epochs) as pbar:
        for i in range(epochs):
            temp = np.zeros(len(train_data_loader))
            for idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()

                out = model(move_to_device(batch, device),)
                out = handle_heterogenous_sizes(batch.y.to(device), out)

                if out.dim() == 2:
                    #TODO: less messy handling of outputs 
                    by = batch.y.view(out.shape[0], out.shape[1]).to(device)
                else:
                    by = batch.y.to(device)
            
                if hasattr(batch, 'mask') and batch.mask is not None:
                    mask = batch.mask.to(device)
                    if mask.sum() == 0:
                        continue  # Skip this batch

                    loss_per_element = loss_func(out, by, reduction='none')
                    loss_per_element = loss_per_element * mask
                    loss_train = loss_per_element.sum() / mask.sum()
                else:
                    # no mask; all targets assumed present
                    loss_train = loss_func(out, by)

                temp[idx] = loss_train.detach().cpu().numpy()

                loss_train.backward()
                optimizer.step()
                #
            loss_train = np.mean(temp)
            train_loss.append(loss_train)

            temp = np.zeros(len(val_data_loader))
            for idx, batch in enumerate(val_data_loader):
                out = model(move_to_device(batch, device),)
                out = handle_heterogenous_sizes(batch.y, out)
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
                    if tuning:
                        pass
                    else:
                        break

            pbar.update(1)
        if early_stopper and not early_stopper.stop and model_name:
            torch.save(model.state_dict(), model_name)
            print(f'Model saved at: {model_name}')

    return train_loss, val_loss


##############################################################################################################
################ Train loop with only jit-friendly operations ################################################
##############################################################################################################

import torch
import torch.nn.functional as F
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from typing import Union, Callable, List
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader, Batch

def train_model_jit(
    model: torch.nn.Module,
    loss_func: Union[Callable, str],
    optimizer: Optimizer,
    train_data_loader: Union[List[Data], DataLoader],
    val_data_loader: Union[List[Data], DataLoader],
    device: str = None,
    epochs: int = 50,
    batch_size: int = 32,
    early_stopper=None,
    scheduler: lr_scheduler._LRScheduler = None,
    tuning: bool = False,
    model_name: str = None,
    model_needs_frag: bool = False,
    net_params: dict = None,
) -> tuple[List[float], List[float]]:
    """
    Training function adapted for the JIT-compiled model, which requires individual tensors as input.
    """
    loss_functions = {
        'mse': F.mse_loss,
        'mae': F.l1_loss
    }

    if isinstance(loss_func, str):
        loss_func = loss_functions[loss_func]

    device = torch.device('cpu') if device is None else torch.device(device)

    exclude_keys = None
    # Exclude fragmentation keys if the model doesn't need them
    if not model_needs_frag:
        if hasattr(train_data_loader, "fragmentation"):
            if train_data_loader.fragmentation is not None:
                exclude_keys = ["frag_graphs", "motif_graphs"]

    if not isinstance(train_data_loader, DataLoader):
        train_data_loader = DataLoader(
            train_data_loader, batch_size=batch_size, exclude_keys=exclude_keys
        )

    if not isinstance(val_data_loader, DataLoader):
        val_data_loader = DataLoader(
            val_data_loader, batch_size=batch_size, exclude_keys=exclude_keys
        )

    model.train()
    train_loss = []
    val_loss = []

    def handle_heterogenous_sizes(y, out):
        if not isinstance(out, torch.Tensor):
            return out
        if y.dim() == out.dim():
            return out
        return out.squeeze()  # Needed for some models

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            temp = np.zeros(len(train_data_loader))
            for idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                # Extract tensors from batch (same as above)
                data_x = batch.x.to(device)
                data_edge_index = batch.edge_index.to(device)
                data_edge_attr = batch.edge_attr.to(device)
                data_batch = batch.batch.to(device)

                # Fragment graphs
                frag_graphs = batch.frag_graphs  # List[Data]
                frag_batch_list = []
                frag_x_list = []
                frag_edge_index_list = []
                frag_edge_attr_list = []
                node_offset = 0
                for i, frag in enumerate(frag_graphs):
                    num_nodes = frag.num_nodes
                    frag_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                    frag_x_list.append(frag.x.to(device))
                    adjusted_edge_index = frag.edge_index + node_offset
                    frag_edge_index_list.append(adjusted_edge_index.to(device))
                    frag_edge_attr_list.append(frag.edge_attr.to(device))
                    node_offset += num_nodes

                frag_x = torch.cat(frag_x_list, dim=0)
                frag_edge_index = torch.cat(frag_edge_index_list, dim=1)
                frag_edge_attr = torch.cat(frag_edge_attr_list, dim=0)
                #frag_batch_1 = torch.cat(frag_batch_list, dim=0)
                frag_batch = Batch.from_data_list(batch.frag_graphs).to(device).batch #moved this computation to training loop
                motif_nodes = Batch.from_data_list(batch.frag_graphs).x
                # Adjusted code:
                junction_graphs = batch.motif_graphs  # List[Data]
                junction_batch_list = []
                junction_x_list = []
                junction_edge_index_list = []
                junction_edge_attr_list = []
                node_offset = 0
                for i, motif in enumerate(junction_graphs):
                    num_nodes = motif.num_nodes
                    junction_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                    junction_x_list.append(motif.x.to(device))
                    adjusted_edge_index = motif.edge_index + node_offset
                    junction_edge_index_list.append(adjusted_edge_index.to(device))
                    junction_edge_attr_list.append(motif.edge_attr.to(device))
                    node_offset += num_nodes

                junction_x = torch.cat(junction_x_list, dim=0)
                junction_edge_index = torch.cat(junction_edge_index_list, dim=1)
                junction_edge_attr = torch.cat(junction_edge_attr_list, dim=0)
                junction_batch = torch.cat(junction_batch_list, dim=0)

                if hasattr(batch, 'global_feats') and batch.global_feats is not None:
                    global_feats = batch.global_feats.to(device)
                else:
                    num_mols = data_batch.max().item() + 1  # Number of molecules in the batch
                    global_feats = torch.zeros((num_mols, 1), device=device)  # Singleton per molecule
                    #global_feats = torch.empty(0).to(device) #can't have optional args in jit so passing this instead

                out = model(
                    data_x,
                    data_edge_index,
                    data_edge_attr,
                    data_batch,
                    frag_x,
                    frag_edge_index,
                    frag_edge_attr,
                    frag_batch,
                    junction_x,
                    junction_edge_index,
                    junction_edge_attr,
                    junction_batch,
                    motif_nodes,
                    global_feats,
                )

                out = handle_heterogenous_sizes(batch.y.to(device), out)

                if out.dim() == 2:
                    #TODO: less messy handling of outputs 
                    by = batch.y.view(out.shape[0], out.shape[1]).to(device)
                else:
                    by = batch.y.to(device)
            
                if hasattr(batch, 'mask') and batch.mask is not None:
                    mask = batch.mask.to(device)
                    if mask.sum() == 0:
                        continue  # Skip this batch

                    loss_per_element = loss_func(out, by, reduction='none')
                    loss_per_element = loss_per_element * mask
                    loss_train = loss_per_element.sum() / mask.sum()
                else:
                    # no mask; all targets assumed present
                    loss_train = loss_func(out, by)

                temp[idx] = loss_train.detach().cpu().numpy()

                loss_train.backward()
                optimizer.step()

            loss_train = np.mean(temp)
            train_loss.append(loss_train)

            # Validation loop
            model.eval()
            temp = np.zeros(len(val_data_loader))
            with torch.no_grad():
                for idx, batch in enumerate(val_data_loader):
                    # Extract tensors from batch (same as above)
                    data_x = batch.x.to(device)
                    data_edge_index = batch.edge_index.to(device)
                    data_edge_attr = batch.edge_attr.to(device)
                    data_batch = batch.batch.to(device)

                    # Fragment graphs
                    frag_graphs = batch.frag_graphs  # List[Data]
                    frag_batch_list = []
                    frag_x_list = []
                    frag_edge_index_list = []
                    frag_edge_attr_list = []
                    node_offset = 0
                    for i, frag in enumerate(frag_graphs):
                        num_nodes = frag.num_nodes
                        frag_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                        frag_x_list.append(frag.x.to(device))
                        adjusted_edge_index = frag.edge_index + node_offset
                        frag_edge_index_list.append(adjusted_edge_index.to(device))
                        frag_edge_attr_list.append(frag.edge_attr.to(device))
                        node_offset += num_nodes

                    frag_x = torch.cat(frag_x_list, dim=0)
                    frag_edge_index = torch.cat(frag_edge_index_list, dim=1)
                    frag_edge_attr = torch.cat(frag_edge_attr_list, dim=0)
                    #frag_batch = torch.cat(frag_batch_list, dim=0)

                    # Junction graphs (motif graphs)
                    motif_graphs = batch.motif_graphs  # List[Data]
                    junction_batch_list = []
                    junction_x_list = []
                    junction_edge_index_list = []
                    junction_edge_attr_list = []
                    # Remove motif_nodes_list if motif_nodes is not available
                    # motif_nodes_list = []
                    node_offset = 0
                    for i, motif in enumerate(motif_graphs):
                        num_nodes = motif.num_nodes
                        junction_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                        junction_x_list.append(motif.x.to(device))
                        adjusted_edge_index = motif.edge_index + node_offset
                        junction_edge_index_list.append(adjusted_edge_index.to(device))
                        junction_edge_attr_list.append(motif.edge_attr.to(device))
                        # If motif_nodes is not available, you can skip this
                        node_offset += num_nodes

                    frag_batch = Batch.from_data_list(batch.frag_graphs).to(device).batch #moved this computation to training loop
                    motif_nodes = Batch.from_data_list(batch.frag_graphs).to(device).x

                    junction_x = torch.cat(junction_x_list, dim=0)
                    junction_edge_index = torch.cat(junction_edge_index_list, dim=1)
                    junction_edge_attr = torch.cat(junction_edge_attr_list, dim=0)
                    junction_batch = torch.cat(junction_batch_list, dim=0)
                    # If motif_nodes is not available, create a placeholder or adjust the model accordingly
                    #motif_nodes = torch.zeros(junction_x.size(0), net_params['frag_dim']).to(device)
                    if hasattr(batch, 'global_feats') and batch.global_feats is not None:
                        global_feats = batch.global_feats.to(device)
                    else:
                        global_feats = torch.empty(0).to(device) #can't have optional args in jit so passing this instead
                    # Forward pass
                    out = model(
                        data_x,
                        data_edge_index,
                        data_edge_attr,
                        data_batch,
                        frag_x,
                        frag_edge_index,
                        frag_edge_attr,
                        frag_batch,
                        junction_x,
                        junction_edge_index,
                        junction_edge_attr,
                        junction_batch,
                        motif_nodes,
                        global_feats,
                    )

                    out = handle_heterogenous_sizes(batch.y.to(device), out)
                    
                    if out.dim() == 2:
                        by = batch.y.view(out.shape[0], out.shape[1]).to(device)
                    else:
                        by = batch.y.to(device)
                    if hasattr(batch, 'mask') and batch.mask is not None:
                        mask = batch.mask.to(device)
                        if mask.sum() == 0:
                            continue  # Skip this batch

                        loss_per_element = loss_func(out, by, reduction='none')
                        loss_per_element = loss_per_element * mask
                        loss_val = loss_per_element.sum() / mask.sum()
                    else:
                        loss_val = loss_func(out, by)
                    temp[idx] = loss_val.detach().cpu().numpy()

            loss_val = np.mean(temp)
            val_loss.append(loss_val)
            model.train()  # Switch back to training mode

            if epoch % 2 == 0:
                pbar.set_description(f"Epoch {epoch}, Training Loss: {loss_train:.3f}, Validation Loss: {loss_val:.3f}")

            if scheduler is not None:
                scheduler.step(loss_val)

            if early_stopper is not None:
                early_stopper(val_loss=loss_val, model=model)
                if early_stopper.stop:
                    #TODO: make into logger instead
                    #print("Early stopping reached with best validation loss: {:.4f}".format(early_stopper.best_score))
                    early_stopper.stop_epoch = epoch - early_stopper.patience
                    if tuning:
                        pass
                    else:
                        break

            pbar.update(1)
        if early_stopper and not early_stopper.stop and model_name:
            torch.save(model.state_dict(), model_name)
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

    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(batch.to(device))
        if out.dim() == 2:
            by = batch.y.view(out.shape[0], out.shape[1]).to(device)
        else:
            by = batch.y.to(device)

        loss_train = loss_func(by, out)

        loss += loss_train.detach().cpu().numpy()
        it += 1.

        loss_train.backward()
        optimizer.step()

        # specifically for the single task ensemble GroupGAT during Hyperparam optimization
        del loss_train, out, by
        torch.cuda.empty_cache()

    return loss/it


def val_epoch(model: torch.nn.Module, loss_func: Callable, val_loader, device: str = None):
    if device is None:
        device = torch.device('cpu')
    model.eval()

    loss = 0.
    it = 0.

    for _, batch in enumerate(val_loader):
        out = model(batch.to(device))
        if out.dim() == 2:
            by = batch.y.view(out.shape[0], out.shape[1]).to(device)
        else:
            by = batch.y.to(device)
        loss_val = loss_func(by, out)
        loss += loss_val.detach().cpu().numpy()
        it += 1.

    return loss/it

##############################################################################################################
#############          Model testing for models with only jit-friendly operations                 ############
##############################################################################################################

def train_epoch_jittable(
    model: torch.nn.Module,
    loss_func: Callable,
    optimizer: torch.optim.Optimizer,
    train_loader,
    device: str = None,
    net_params: dict = None,
):
    """
    see `train_epoch` for args and easily readable logic. 
    This version is for jittable models where all arguments 
    need to be passed in explicitly
    """
    if device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    model.train()
    total_loss = 0.0
    total_iters = 0.0

    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # Extract tensors from batch
        data_x = batch.x.to(device)
        data_edge_index = batch.edge_index.to(device)
        data_edge_attr = batch.edge_attr.to(device)
        data_batch = batch.batch.to(device)

        # Fragment graphs
        frag_batch_data = Batch.from_data_list(batch.frag_graphs).to(device)
        frag_x = frag_batch_data.x
        frag_edge_index = frag_batch_data.edge_index
        frag_edge_attr = frag_batch_data.edge_attr
        frag_batch = frag_batch_data.batch

        # Junction graphs (motif graphs)
        junction_batch_data = Batch.from_data_list(batch.motif_graphs).to(device)
        junction_x = junction_batch_data.x
        junction_edge_index = junction_batch_data.edge_index
        junction_edge_attr = junction_batch_data.edge_attr
        junction_batch = junction_batch_data.batch

        # Motif nodes
        motif_nodes = frag_batch_data.x  # Assuming motif nodes are the same as fragment node features

        # Handle global_feats
        if hasattr(batch, 'global_feats') and batch.global_feats is not None:
            global_feats = batch.global_feats.to(device)
        else:
            global_feats = torch.empty(0).to(device)  # Can't have optional args in JIT, so pass an empty tensor

        # Forward pass
        out = model(
            data_x,
            data_edge_index,
            data_edge_attr,
            data_batch,
            frag_x,
            frag_edge_index,
            frag_edge_attr,
            frag_batch,
            junction_x,
            junction_edge_index,
            junction_edge_attr,
            junction_batch,
            motif_nodes,
            global_feats,
        )

        # Compute loss
        if out.dim() == 2:
            by = batch.y.view(out.shape[0], out.shape[1]).to(device)
        else:
            by = batch.y.to(device)

        loss_train = loss_func(by, out)

        # Backward and optimize
        loss_train.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss_train.detach().cpu().numpy()
        total_iters += 1.0

    return total_loss / total_iters


def val_epoch_jittable(
    model: torch.nn.Module,
    loss_func: Callable,
    val_loader,
    device: str = None,
    net_params: dict = None,
):
    """
    see `val_epoch` for args and easily readable logic. 
    This version is for jittable models where all arguments 
    need to be passed in explicitly
    """
    if device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    model.eval()
    total_loss = 0.0
    total_iters = 0.0

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            # Extract tensors from batch
            data_x = batch.x.to(device)
            data_edge_index = batch.edge_index.to(device)
            data_edge_attr = batch.edge_attr.to(device)
            data_batch = batch.batch.to(device)

            # Fragment graphs
            frag_batch_data = Batch.from_data_list(batch.frag_graphs).to(device)
            frag_x = frag_batch_data.x
            frag_edge_index = frag_batch_data.edge_index
            frag_edge_attr = frag_batch_data.edge_attr
            frag_batch = frag_batch_data.batch

            # Junction graphs (motif graphs)
            junction_batch_data = Batch.from_data_list(batch.motif_graphs).to(device)
            junction_x = junction_batch_data.x
            junction_edge_index = junction_batch_data.edge_index
            junction_edge_attr = junction_batch_data.edge_attr
            junction_batch = junction_batch_data.batch

            # Motif nodes
            motif_nodes = frag_batch_data.x  # Assuming motif nodes are the same as fragment node features

            # Handle global_feats
            if hasattr(batch, 'global_feats') and batch.global_feats is not None:
                global_feats = batch.global_feats.to(device)
            else:
                global_feats = torch.empty(0).to(device)  # Can't have optional args in JIT, so pass an empty tensor

            # Forward pass
            out = model(
                data_x,
                data_edge_index,
                data_edge_attr,
                data_batch,
                frag_x,
                frag_edge_index,
                frag_edge_attr,
                frag_batch,
                junction_x,
                junction_edge_index,
                junction_edge_attr,
                junction_batch,
                motif_nodes,
                global_feats,
            )

            # Compute loss
            if out.dim() == 2:
                by = batch.y.view(out.shape[0], out.shape[1]).to(device)
            else:
                by = batch.y.to(device)

            loss_val = loss_func(by, out)

            # Accumulate loss
            total_loss += loss_val.detach().cpu().numpy()
            total_iters += 1.0

    return total_loss / total_iters



##############################################################################################################
################################# Model testing ##############################################################
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
        Mean test loss over the test set batches.

    """


    device = torch.device('cpu') if device is None else device

    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader([data for data in test_data_loader], batch_size = batch_size)

    model.eval()

    with tqdm(total = len(test_data_loader)) as pbar:

        for idx, batch in enumerate(test_data_loader):
            # TODO: Broaden use of return_latents
            out = model(batch.to(device))
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

    if return_latents:
        return preds, latents
    return preds

##############################################################################################################
################################# JIttable Model Testing #####################################################
##############################################################################################################


def test_model_jit(
    model: torch.nn.Module,
    test_data_loader: Union[List, Data, DataLoader],
    device: str = None,
    batch_size: int = 32,
    return_latents: bool = False,
    model_needs_frag: bool = False,
    net_params: dict = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Auxiliary function to test a trained JIT-compiled model and return the predictions as well as the latent node
    representations. Can initialize DataLoaders if only a list of Data objects is given.

    Notes
    -----
    This function is designed for JIT-compiled models that require individual tensors as input.

    Parameters
    ----------
    model : torch.nn.Module
        The JIT-compiled model to be tested.
    test_data_loader : list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the test graphs.
    device : str, optional
        Torch device to be used ('cpu', 'cuda', or 'mps'). Default is 'cpu'.
    batch_size : int, optional
        Batch size of the DataLoader if not given directly. Default is 32.
    return_latents : bool, optional
        Determines if the latents should be returned. **If used, the model must include `return_lats` parameter**.
        Default is False.
    model_needs_frag : bool, optional
        Whether the model needs fragment graphs or not. Default is False.
    net_params : dict, optional
        A dictionary containing network parameters, used for dimension matching if necessary.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Predictions, and optionally latents if `return_latents` is True.
    """
    device = torch.device('cpu') if device is None else torch.device(device)

    # Exclude fragmentation keys if the model doesn't need them
    exclude_keys = None
    if not model_needs_frag:
        exclude_keys = ["frag_graphs", "motif_graphs"]

    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader([data for data in test_data_loader], batch_size = batch_size)

    model.eval()

    with torch.no_grad():
        with tqdm(total=len(test_data_loader)) as pbar:
            preds_list = []
            if return_latents:
                latents_list = []
            for idx, batch in enumerate(test_data_loader):
                # Extract tensors from batch
                data_x = batch.x.to(device)
                data_edge_index = batch.edge_index.to(device)
                data_edge_attr = batch.edge_attr.to(device)
                data_batch = batch.batch.to(device)

                # Fragment graphs
                frag_graphs = batch.frag_graphs  # List[Data]
                frag_batch_list = []
                frag_x_list = []
                frag_edge_index_list = []
                frag_edge_attr_list = []
                node_offset = 0
                for i, frag in enumerate(frag_graphs):
                    num_nodes = frag.num_nodes
                    frag_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                    frag_x_list.append(frag.x.to(device))
                    adjusted_edge_index = frag.edge_index + node_offset
                    frag_edge_index_list.append(adjusted_edge_index.to(device))
                    frag_edge_attr_list.append(frag.edge_attr.to(device))
                    node_offset += num_nodes

                frag_x = torch.cat(frag_x_list, dim=0)
                frag_edge_index = torch.cat(frag_edge_index_list, dim=1)
                frag_edge_attr = torch.cat(frag_edge_attr_list, dim=0)
                frag_batch = torch.cat(frag_batch_list, dim=0)

                # Junction graphs (motif graphs)
                motif_graphs = batch.motif_graphs  # List[Data]
                junction_batch_list = []
                junction_x_list = []
                junction_edge_index_list = []
                junction_edge_attr_list = []
                node_offset = 0
                for i, motif in enumerate(motif_graphs):
                    num_nodes = motif.num_nodes
                    junction_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                    junction_x_list.append(motif.x.to(device))
                    adjusted_edge_index = motif.edge_index + node_offset
                    junction_edge_index_list.append(adjusted_edge_index.to(device))
                    junction_edge_attr_list.append(motif.edge_attr.to(device))
                    node_offset += num_nodes

                junction_x = torch.cat(junction_x_list, dim=0)
                junction_edge_index = torch.cat(junction_edge_index_list, dim=1)
                junction_edge_attr = torch.cat(junction_edge_attr_list, dim=0)
                junction_batch = torch.cat(junction_batch_list, dim=0)

                frag_batch = Batch.from_data_list(batch.frag_graphs).to(device).batch #moved this computation to training loop
                motif_nodes = Batch.from_data_list(batch.frag_graphs).to(device).x

                if hasattr(batch, 'global_feats'):
                    global_feats = batch.global_feats.to(device)
                else:
                    global_feats = torch.empty(0).to(device)

                # Forward pass
                if return_latents:
                    # Assuming the model's forward method supports `return_lats` parameter
                    out, lat = model(
                        data_x,
                        data_edge_index,
                        data_edge_attr,
                        data_batch,
                        frag_x,
                        frag_edge_index,
                        frag_edge_attr,
                        frag_batch,
                        junction_x,
                        junction_edge_index,
                        junction_edge_attr,
                        junction_batch,
                        motif_nodes,
                        global_feats
                    )
                    lat = lat.detach().cpu()
                    latents_list.append(lat)
                else:
                    out = model(
                        data_x,
                        data_edge_index,
                        data_edge_attr,
                        data_batch,
                        frag_x,
                        frag_edge_index,
                        frag_edge_attr,
                        frag_batch,
                        junction_x,
                        junction_edge_index,
                        junction_edge_attr,
                        junction_batch,
                        motif_nodes,
                        global_feats
                    )

                out = out.detach().cpu()
                preds_list.append(out)

                pbar.update(1)
                
    preds = torch.cat(preds_list, dim=0)
    if return_latents:
        latents = torch.cat(latents_list, dim=0)
        return preds, latents
    else:
        return preds
    # Concatenate predictions and latents if needed to write to a csv or smth
        # preds = torch.cat(preds_list, dim=0)
        # if output_csv is not None:
        #     # Rescale predictions if mean and std are provided
        #     if mean is not None and std is not None:
        #         # Ensure mean and std are tensors of correct shape
        #         mean_tensor = torch.tensor(mean, dtype=preds.dtype).view(1, -1)
        #         std_tensor = torch.tensor(std, dtype=preds.dtype).view(1, -1)
        #         preds_rescaled = preds * std_tensor + mean_tensor
        #     else:
        #         preds_rescaled = preds
        #     # Convert to numpy array
        #     preds_array = preds_rescaled.numpy()
        #     # Create DataFrame with SMILES and predictions
        #     preds_df = pd.DataFrame(preds_array)
        #     preds_df.insert(0, 'SMILES', smiles_list)
        #     # Save to CSV
        #     preds_df.to_csv(output_csv, index=False)
        # if return_latents:
        #     latents = torch.cat(latents_list, dim=0)
        #     return preds, latents
        # else:
        #     return preds
    
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




def pred_metric(
    prediction: Union[Tensor, ndarray],
    target: Union[Tensor, ndarray],
    metrics: Union[str, List[str]] = 'mse',
    print_out: bool = True,
    rescale_data: 'DataSet' = None
) -> List[float]:
    """
    A function to evaluate continuous predictions compared to targets with different metrics. It can
    take either Tensors or ndarrays and will automatically convert them to the correct format. Partly makes use of
    sklearn and their implementations. The options for metrics are:

    * ``MSE``: Mean Squared Error
    * ``RMSE``: Root Mean Squared Error
    * ``SSE``: Sum of Squared Errors
    * ``MAE``: Mean Absolute Error
    * ``R2``: R-squared Error
    * ``MRE``: Mean Relative Error
    * ``MDAPE``: Median Absolute Percentage Error
    * ``MARE``: Mean Absolute Relative Error
    * ``Pearson``: Pearson Correlation Coefficient

    .. math::
        \frac{1}{N}\sum\limits_{i=1}^{N}\frac{|y_{i}-f(x_{i})|}{|y_{i}|}\cdot100

    **If a list of metrics is given, then a list of equal length is returned.**

    Parameters
    -----------
    prediction: Tensor or ndarray
        A prediction array or tensor generated by some sort of model.
    target: Tensor or ndarray
        The target array or tensor corresponding to the prediction.
    metrics: str or list[str]
        A string or a list of strings specifying what metrics should be returned. The options are:
        [``mse``, ``rmse``, ``sse``, ``mae``, ``r2``, ``mre``, ``mdape``, ``mape``, ``mare``, ``pearson``] or 'all' for every option. Default: 'mse'
    print_out: bool
        Will print out formatted results if True. Default: True
    rescale_data: DataSet, optional
        An instance of DataSet used to rescale the prediction and target data. Default: None

    Returns
    --------
    list[float]
        A list equal to the number of metrics specified containing the corresponding results.
    """

    # Convert Tensors to numpy arrays if necessary
    if isinstance(prediction, Tensor):
        prediction = prediction.cpu().detach().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().detach().numpy()

    # Ensure metrics is a list
    if not isinstance(metrics, list) and metrics != 'all':
        metrics = [metrics]

    # Rescale data if rescale_data is provided
    if rescale_data is not None:
        prediction = rescale_data.rescale_data(prediction)
        target = rescale_data.rescale_data(target)

    # If metrics is 'all', include all available metrics
    if metrics == 'all':
        metrics = ['mse', 'rmse', 'sse', 'mae', 'r2', 'mre', 'mdape', 'mape', 'mare', 'pearson']

    results = {}
    prints = []
    epsilon = 1e-12  # Small value to prevent division by zero

    # Flatten the arrays for certain metrics
    target_flat = target.flatten()
    prediction_flat = prediction.flatten()

    # Define a helper function to calculate MARE
    def calc_MARE(ym, yp):
        RAE = []
        pstd = np.std(ym)
        for i in range(len(ym)):
            if -0.1 <= ym[i] <= 0.1:
                RAE.append(abs(ym[i] - yp[i]) / (pstd + epsilon) * 100)
            else:
                RAE.append(abs(ym[i] - yp[i]) / (abs(ym[i]) + epsilon) * 100)
        mare = np.mean(RAE)
        return mare

    for metric_ in metrics:
        metric_lower = metric_.lower()
        if metric_lower == 'mse':
            mse = mean_squared_error(target, prediction)
            results['mse'] = mse
            prints.append(f'MSE: {mse:.3f}')
        elif metric_lower == 'rmse':
            mse = mean_squared_error(target, prediction)
            rmse = np.sqrt(mse)
            results['rmse'] = rmse
            prints.append(f'RMSE: {rmse:.3f}')
        elif metric_lower == 'sse':
            sse = np.sum((target - prediction) ** 2)
            results['sse'] = sse
            prints.append(f'SSE: {sse:.3f}')
        elif metric_lower == 'mae':
            mae = mean_absolute_error(target, prediction)
            results['mae'] = mae
            prints.append(f'MAE: {mae:.3f}')
        elif metric_lower == 'r2':
            r2 = r2_score(target, prediction)
            results['r2'] = r2
            prints.append(f'R2: {r2:.3f}')
        elif metric_lower == 'pearson':
            if np.std(target_flat) == 0 or np.std(prediction_flat) == 0:
                pearson_corr = 0.0  # Avoid division by zero
            else:
                pearson_corr = np.corrcoef(target_flat, prediction_flat)[0, 1]
            results['pearson'] = pearson_corr
            prints.append(f'Pearson Corr Coeff: {pearson_corr:.3f}')
        elif metric_lower == 'mape':
            mape = mean_absolute_percentage_error(target, prediction) * 100  # Convert to percentage
            results['mape'] = mape
            prints.append(f'MAPE: {mape:.3f}%')
        elif metric_lower == 'mre':
            mre = np.mean(np.abs((target - prediction) / (np.abs(target) + epsilon))) * 100
            results['mre'] = mre
            prints.append(f'MRE: {mre:.3f}%')
            if mre > 100:
                median_re = np.median(np.abs((target - prediction) / (np.abs(target) + epsilon))) * 100
                prints.append(f'Median Relative Error: {median_re:.3f}%')
        elif metric_lower == 'mdape':
            mdape = np.median(np.abs((target - prediction) / (np.abs(target) + epsilon))) * 100
            results['mdape'] = mdape
            prints.append(f'MDAPE: {mdape:.3f}%')
        elif metric_lower == 'mare':
            mare = calc_MARE(target_flat, prediction_flat)
            results['mare'] = mare
            prints.append(f'MARE: {mare:.3f}%')
        else:
            raise ValueError(f"Unsupported metric: {metric_}")

    if print_out:
        for out in prints:
            print(out)

    return results

#######################################################################################################
#################################### Learning curve visualizer ########################################
#######################################################################################################
class learning_curve_producer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def update(self, train_loss, val_loss=None):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def print_losses(self):
        print('Training losses:', self.train_losses)
        if self.val_losses:
            print('Validation losses:', self.val_losses)

    def display_learning_curve(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()

#######################################################################################################
#################################### General tools ####################################################
#######################################################################################################


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







