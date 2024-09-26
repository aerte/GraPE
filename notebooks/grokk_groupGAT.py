from grape_chem.models import AFP
from grape_chem.models import GroupGAT
from grape_chem.utils import DataSet, train_model, EarlyStopping, split_data, test_model, pred_metric, return_hidden_layers, set_seed, JT_SubGraph, FragmentGraphDataSet
from grape_chem.datasets import FreeSolv 
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Data, Batch
from tqdm import tqdm
from typing import Union, List
from torch import Tensor
import os
## Install GraPE with: pip install "git+https://github.com/aerte/GraPE.git#subdirectory=python"

def standardize(x, mean, std):
    return (x - mean) / std

##########################################################################################
#####################    Data Input Region  ##############################################
##########################################################################################

set_seed(42)

# Hyperparameters
epochs = 1000
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.00126
weight_decay = 1e-4
mlp_layers = 2
atom_layers = 3
mol_layers = 3


# Change to your own specifications
root = './env/Vc_cace.xlsx'
sheet_name = 'Melting Point'

df = pd.read_excel(root,)#.iloc[:25] 
smiles = df['SMILES'].to_numpy()
target = df['Target'].to_numpy()

#specific to one xlsx with a "Tag" column
tags = df['Tag'].to_numpy()
unique_tags = np.unique(tags)
tag_to_int = {'Train': 0, 'Val': 1, 'Test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

### Global feature from sheet, uncomment
#global_feats = df['Global Feats'].to_numpy()

#### REMOVE, just for testing ####
global_feats = np.random.randn(len(smiles))

############ We need to standardize BEFORE loading it into a DataSet #############
mean_target, std_target = np.mean(target), np.std(target)
target = standardize(target, mean_target, std_target)
mean_global_feats, std_global_feats = np.mean(global_feats), np.std(global_feats)
global_feats = standardize(global_feats, mean_global_feats, std_global_feats)


########################## fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("initializing frag...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme)
frag_dim = fragmentation.frag_dim
print("done.")


########################### FreeSolv ###################################################
#data = FreeSolv(fragmentation=fragmentation)
########################################################################################

######################## QM9 / testing /excel ##########################################
data = DataSet(smiles=smiles, target=target, global_features=None, filter=True, fragmentation=fragmentation)
########################################################################################


#train_set, val_set, _ = data.split_and_scale(scale=True, split_type='random')
train, val, test = split_data(data, split_type='custom', custom_split=custom_split,)
############################################################################################
############################################################################################
############################################################################################

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# num_global_feats is the dimension of global features per observation
mlp = return_hidden_layers(mlp_layers)
net_params = {
              "device": device, #shouldn't be passed in in this way, but best we have for now  
              "num_atom_type": 44, # == node_in_dim TODO: check matches with featurizer or read from featurizer
              "num_bond_type": 12, # == edge_in_dim
              "dropout": 0.0,
              "MLP_layers":mlp_layers,
              "frag_dim": frag_dim,
              "final_dropout": 0.119,
            # for origins:
              "num_heads": 1,
            # for AFP:
              "node_in_dim": 44, 
              "edge_in_dim": 12, 
              "num_global_feats":1, 
              "hidden_dim": hidden_dim, #Important: check matches with `L1_hidden_dim`
              "mlp_out_hidden": mlp, 
              "num_layers_atom": atom_layers, 
              "num_layers_mol": mol_layers,
            # for channels:
              "L1_layers_atom": 4, #L1_layers
              "L1_layers_mol": 1,  #L1_depth
              "L1_dropout": 0.142,

              "L2_layers_atom": 2, #L2_layers
              "L2_layers_mol": 3,  #2_depth
              "L2_dropout": 0.255,

              "L3_layers_atom": 1, #L3_layers
              "L3_layers_mol": 4,  #L3_depth
              "L3_dropout": 0.026,

              "L1_hidden_dim": 247,
              "L2_hidden_dim": 141,
              "L3_hidden_dim": 47,
              }
model = GroupGAT.GCGAT_v4pro(net_params)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_Stopper = EarlyStopping(patience=patience, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, min_lr=1e-09,
                                           patience=patience)

loss_func = torch.nn.functional.l1_loss

model.to(device)

# Define model filename
model_filename = 'gcgat_latest.pth'

# Check if the model file exists
if os.path.exists(model_filename):
    print(f"Model file '{model_filename}' found. Loading the trained model.")
    # Load the model state dict
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
else:
    print(f"No trained model found at '{model_filename}'. Proceeding to train the model.")
    # Train the model
    train_model(model=model, loss_func=loss_func, optimizer=optimizer, train_data_loader=train,
                val_data_loader=val, epochs=epochs, device=device, batch_size=batch_size, scheduler=scheduler, model_needs_frag=True)
    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to '{model_filename}'.")

####### Generating prediction tensor for the TEST set (Not rescaled) #########

pred = test_model(model=model, test_data_loader=test, device=device, batch_size=batch_size) #TODO: make it able to take a loss func
pred_metric(prediction=pred, target=test.y, metrics='all', print_out=True)

# ---------------------------------------------------------------------------------------



####### Example for rescaling the MAE prediction ##########

test_mae = pred_metric(prediction=pred, target=test.y, metrics='mae', print_out=False)['mae']
#test_mae_rescaled = test_mae * std_target + mean_target #TODO: add rescaling to the 
#print(f'Rescaled MAE for the test set {test_mae_rescaled:.3f}')

# ---------------------------------------------------------------------------------------


####### Example for overall evaluation of the MAE #########

train_preds = test_model(model=model, test_data_loader=train, device=device) #TODO
val_preds = test_model(model=model, test_data_loader=val, device=device)

train_mae = pred_metric(prediction=train_preds, target=train.y, metrics='mae', print_out=False)['mae']
val_mae = pred_metric(prediction=val_preds, target=val.y, metrics='mae', print_out=False)['mae']

#overall_mae = (train_mae+val_mae+test_mae)/3 * std_target + mean_target
#print(f'Rescaled overall MAE {overall_mae:.3f}')

##############################################################################################################
################ Train loop with grokking   ##################################################################
##############################################################################################################


from data_files.grokfast import gradfilter_ma, gradfilter_ema

def train_model_grokkfast(model: torch.nn.Module, loss_func: Union[Callable,str], optimizer: torch.optim.Optimizer,
                    train_data_loader: Union[list, Data, DataLoader], val_data_loader: Union[list, Data, DataLoader],
                    device: str = None, epochs: int = 50, batch_size: int = 32,
                    early_stopper: EarlyStopping = None, scheduler: lr_scheduler = None,
                    tuning: bool = False, model_name:str = None, model_needs_frag : bool = False,
                    grokfast_type: str = 'ema', alpha: float = 0.9, lamb: float = 0.1, window_size: int = 10,) -> tuple[list,list]:
    """Auxiliary function to train and test a given model using Grokfast and return the (training, test) losses.
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
        Will turn off the early stopping, meant for hyperparameter optimization.
    model_name:str
        If given, it will be used to save it if early stopping did not set in. Default: None
    model_needs_frag : bool
        Whether the model needs fragmentation data or not.
    grokfast_type: str
        Type of Grokfast to use, either 'ema' or 'ma'. Default: 'ema'
    alpha: float
        Alpha parameter for gradfilter_ema. Default: 0.9
    lamb: float
        Lambda parameter for gradfilter functions. Default: 0.1
    window_size: int
        Window size parameter for gradfilter_ma. Default: 10

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
    # In some cases, this DataLoader shouldn't batch the graphs that result from fragmentations with the rest
    if not model_needs_frag:
        if hasattr(train_data_loader, "fragmentation"):
            if train_data_loader.fragmentation is not None:
                exclude_keys = ["frag_graphs", "motif_graphs"]
    
    if not isinstance(train_data_loader, DataLoader):
        train_data_loader = DataLoader(train_data_loader, batch_size=batch_size, exclude_keys=exclude_keys)

    if not isinstance(val_data_loader, DataLoader):
        val_data_loader = DataLoader(val_data_loader, batch_size=batch_size, exclude_keys=exclude_keys)

    model.train()
    grads = None  # Initialize grads for Grokfast
    train_loss = []
    val_loss = []

    def handle_heterogenous_sizes(y, out):
        """
        Adjusts the output to match the target size if necessary.
        """
        if not isinstance(out, torch.Tensor):
            return out
        if y.dim() == out.dim():
            return out
        return out.squeeze()  # Needed for some models

    def move_to_device(data, device):
        """
        Moves data to the specified device, handling nested structures.
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, list):
            return [move_to_device(item, device) for item in data]
        elif isinstance(data, dict):
            return {key: move_to_device(value, device) for key, value in data.items()}
        else:
            return data

    with tqdm(total=epochs) as pbar:
        for i in range(epochs):
            temp = np.zeros(len(train_data_loader))
            for idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()

                out = model(move_to_device(batch, device))
                out = handle_heterogenous_sizes(batch.y, out)

                loss_train = loss_func(batch.y, out)
                temp[idx] = loss_train.detach().cpu().numpy()

                loss_train.backward()

                # Apply Grokfast gradient filtering
                if grokfast_type == 'ema':
                    grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)
                elif grokfast_type == 'ma':
                    grads = gradfilter_ma(model, grads=grads, window_size=window_size, lamb=lamb)
                else:
                    raise ValueError(f"Unknown grokfast_type: {grokfast_type}")

                optimizer.step()

            loss_train = np.mean(temp)
            train_loss.append(loss_train)

            temp = np.zeros(len(val_data_loader))
            for idx, batch in enumerate(val_data_loader):
                out = model(move_to_device(batch, device))
                out = handle_heterogenous_sizes(batch.y, out)
                temp[idx] = loss_func(batch.y, out).detach().cpu().numpy()

            loss_val = np.mean(temp)
            val_loss.append(loss_val)

            if i % 2 == 0:
                pbar.set_description(f"epoch={i}, training loss= {loss_train:.3f}, validation loss= {loss_val:.3f}")

            if scheduler is not None:
                scheduler.step(loss_val)

            if early_stopper is not None:
                early_stopper(val_loss=loss_val, model=model)
                if early_stopper.stop:
                    early_stopper.stop_epoch = i - early_stopper.patience
                    if tuning:
                        pass
                    else:
                        break

            pbar.update(1)
        if early_stopper and not early_stopper.stop and model_name:
            torch.save(model.state_dict(), model_name)
            print(f'Model saved at: {model_name}')

    return train_loss, val_loss