
from grape_chem.models import GroupGAT_jittable
from grape_chem.utils import DataSet, train_model_jit, EarlyStopping, split_data, test_model_jit, pred_metric, return_hidden_layers, set_seed, JT_SubGraph, FragmentGraphDataSet
from grape_chem.datasets import FreeSolv 
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Data, Batch
from tqdm import tqdm
from typing import Union, List, Tuple
from torch import Tensor
import os

## Install GraPE with: pip install "git+https://github.com/aerte/GraPE.git#subdirectory=python"

def standardize(x, mean, std):
    return (x - mean) / std

##########################################################################################
#####################    Data Input Region  ##############################################
##########################################################################################

set_seed(42)

# Change to your own specifications
root = './env/pka_dataset.xlsx'
sheet_name = ''

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
data = DataSet(smiles=smiles, target=target, global_features=None, filter=True, fragmentation=fragmentation, custom_split=custom_split)
custom_split = data.custom_split #messy but it gets recomputed in this way
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


# Hyperparameters
epochs = 600
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.00126
weight_decay = 0.003250012
mlp_layers = 4
atom_layers = 3
mol_layers = 3

# num_global_feats is the dimension of global features per observation
mlp = return_hidden_layers(mlp_layers)
net_params = {
              "device": device, #shouldn't be passed in in this way, but best we have for now  
              "num_atom_type": 44, # == node_in_dim TODO: check matches with featurizer or read from featurizer
              "num_bond_type": 12, # == edge_in_dim
              "dropout": 0.0,
              "MLP_layers":mlp_layers,
              "frag_dim": frag_dim,
              "final_dropout": 0.257507,
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
              "L1_layers_atom": 3, #L1_layers
              "L1_layers_mol": 3,  #L1_depth
              "L1_dropout": 0.370796,

              "L2_layers_atom": 3, #L2_layers
              "L2_layers_mol": 2,  #2_depth
              "L2_dropout": 0.056907,

              "L3_layers_atom": 1, #L3_layers
              "L3_layers_mol": 4,  #L3_depth
              "L3_dropout": 0.137254,

              "L1_hidden_dim": 125,
              "L2_hidden_dim": 155,
              "L3_hidden_dim": 64,
              }
model = torch.jit.script(GroupGAT_jittable.GCGAT_v4pro_jit(net_params))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_Stopper = EarlyStopping(patience=100, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=1e-9,
                                           patience=30)

loss_func = torch.nn.functional.mse_loss

model.to(device)

model_filename = 'gcgat_jitted_latest.pth'

if os.path.exists(model_filename):
    print(f"Model file '{model_filename}' found. Loading the trained model.")
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
else:
    print(f"No trained model found at '{model_filename}'. Proceeding to train the model.")
    train_model_jit(model=model, loss_func=loss_func, optimizer=optimizer, train_data_loader=train,
                val_data_loader=val, epochs=epochs, device=device, batch_size=batch_size, scheduler=scheduler, early_stopper=early_Stopper, model_needs_frag=True, net_params=net_params)

    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to '{model_filename}'.")

####### Generating prediction tensor for the TEST set (Not rescaled) #########
pred = test_model_jit(model=model, test_data_loader=test, device=device, batch_size=batch_size) #TODO: make it able to take a loss func
pred_metric(prediction=pred, target=test.y, metrics='all', print_out=True)

# ---------------------------------------------------------------------------------------



####### Example for rescaling the MAE prediction ##########

test_mae = pred_metric(prediction=pred, target=test.y, metrics='mae', print_out=False)['mae']
#test_mae_rescaled = test_mae * std_target + mean_target #TODO: add rescaling to the 
#print(f'Rescaled MAE for the test set {test_mae_rescaled:.3f}')

# ---------------------------------------------------------------------------------------


####### Example for overall evaluation of the MAE #########

train_preds = test_model_jit(model=model, test_data_loader=train, device=device) #TODO
val_preds = test_model_jit(model=model, test_data_loader=val, device=device)

train_mae = pred_metric(prediction=train_preds, target=train.y, metrics='mae', print_out=False)['mae']
val_mae = pred_metric(prediction=val_preds, target=val.y, metrics='mae', print_out=False)['mae']

#overall_mae = (train_mae+val_mae+test_mae)/3 * std_target + mean_target
#print(f'Rescaled overall MAE {overall_mae:.3f}')

# Modify the test_model function to return both predictions and targets
def test_model_with_parity(model: torch.nn.Module, test_data_loader: Union[list, Data],
               device: str = None, batch_size: int = 32, return_latents: bool = False, model_needs_frag: bool = False) -> (
        Union[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]):
    """Auxiliary function to test a trained model and return the predictions as well as the targets and optional latent node
    representations. If a loss function is specified, then it will also return a list containing the testing losses.
    Can initialize DataLoaders if only list of Data objects are given.

    Parameters
    ------------
    model: torch.nn.Module
        Model that will be tested. Has to be a torch Module.
    test_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the test graphs.
    device: str
        Torch device to be used ('cpu','cuda' or 'mps'). Default: 'cpu'
    batch_size: int
        Batch size of the DataLoader if not given directly. Default: 32
    return_latents: bool
        Decides if the latents should be returned. **If used, the model must include return_latent statement**. Default:
        False
    model_needs_frag: bool
        Indicates whether the model requires fragment information.

    Returns
    ---------
    Tuple[Tensor, Tensor] or Tuple[Tensor, Tensor, Tensor]
        Returns predictions and targets, and optionally latents if return_latents is True.
    """

    device = torch.device('cpu') if device is None else device

    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader([data for data in test_data_loader], batch_size = batch_size)

    model.eval()

    with tqdm(total=len(test_data_loader)) as pbar:

        for idx, batch in enumerate(test_data_loader):
            batch = batch.to(device)
            # Handling models that require fragment information
            if model_needs_frag:
                if return_latents:
                    out, lat = model(batch, return_lats=True)
                else:
                    out = model(batch)
            else:
                if return_latents:
                    out, lat = model(batch, return_lats=True)
                else:
                    out = model(batch)

            # Collect targets
            target = batch.y

            # Concatenate predictions, targets, and latents
            if idx == 0:
                preds = out
                targets = target
                if return_latents:
                    latents = lat
            else:
                preds = torch.cat([preds, out], dim=0)
                targets = torch.cat([targets, target], dim=0)
                if return_latents:
                    latents = torch.cat([latents, lat], dim=0)

            pbar.update(1)

    if return_latents:
        return preds, targets, latents
    return preds, targets


def test_model_jit_with_parity(
    model: torch.nn.Module,
    test_data_loader: Union[List,], #Union[List, DataLoader],
    device: str = None,
    batch_size: int = 32,
    return_latents: bool = False,
    model_needs_frag: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Auxiliary function to test a trained JIT-compiled model and return the predictions as well as the targets and optional latent node
    representations. Can initialize DataLoaders if only list of Data objects are given.

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

    Returns
    -------
    Tuple[Tensor, Tensor] or Tuple[Tensor, Tensor, Tensor]
        Returns predictions and targets, and optionally latents if `return_latents` is True.
    """

    device = torch.device('cpu') if device is None else torch.device(device)
    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader([data for data in test_data_loader], batch_size=batch_size)

    model.eval()

    with torch.no_grad():
        preds_list = []
        targets_list = []
        if return_latents:
            latents_list = []
        with tqdm(total=len(test_data_loader)) as pbar:
            for idx, batch in enumerate(test_data_loader):
                # Move batch to device
                batch = batch.to(device)

                if model_needs_frag:
                    # Extract necessary tensors for models that require fragment information
                    data_x = batch.x
                    data_edge_index = batch.edge_index
                    data_edge_attr = batch.edge_attr
                    data_batch = batch.batch

                    # Fragment graphs
                    frag_graphs = batch.frag_graphs  # List[Data]
                    frag_batch = Batch.from_data_list(frag_graphs).batch.to(device)
                    frag_x = Batch.from_data_list(frag_graphs).x.to(device)
                    frag_edge_index = Batch.from_data_list(frag_graphs).edge_index.to(device)
                    frag_edge_attr = Batch.from_data_list(frag_graphs).edge_attr.to(device)

                    # Junction graphs (motif graphs)
                    motif_graphs = batch.motif_graphs  # List[Data]
                    junction_batch = Batch.from_data_list(motif_graphs).batch.to(device)
                    junction_x = Batch.from_data_list(motif_graphs).x.to(device)
                    junction_edge_index = Batch.from_data_list(motif_graphs).edge_index.to(device)
                    junction_edge_attr = Batch.from_data_list(motif_graphs).edge_attr.to(device)

                    motif_nodes = frag_x  # Assuming motif nodes are fragment node features

                    if hasattr(batch, 'global_feats'):
                        global_feats = batch.global_feats
                    else:
                        global_feats = torch.empty((data_x.size(0), 1), dtype=torch.float32).to(device)

                    global_feats = global_feats.to(device)
                    
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
                            motif_nodes
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
                            motif_nodes
                        )
                else:
                    # For models that do not need fragment information
                    if return_latents:
                        out, lat = model(batch, return_lats=True)
                        lat = lat.detach().cpu()
                        latents_list.append(lat)
                    else:
                        out = model(batch)

                # Collect predictions and targets
                preds_list.append(out.detach().cpu())
                targets_list.append(batch.y.detach().cpu())

                pbar.update(1)

    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    if return_latents:
        latents = torch.cat(latents_list, dim=0)
        return preds, targets, latents
    else:
        return preds, targets

####### Generating predictions and targets for all datasets #########

# Replace 'train_loader', 'val_loader', 'test_loader' with your actual data loaders
train_preds, train_targets = test_model_jit_with_parity(
    model=model,
    test_data_loader=train,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)

val_preds, val_targets = test_model_jit_with_parity(
    model=model,
    test_data_loader=val,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)

test_preds, test_targets = test_model_jit_with_parity(
    model=model,
    test_data_loader=test,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)

####### Creating Parity Plot #########

# Convert tensors to numpy arrays
train_preds_np = train_preds.cpu().numpy()
train_targets_np = train_targets.cpu().numpy()
val_preds_np = val_preds.cpu().numpy()
val_targets_np = val_targets.cpu().numpy()
test_preds_np = test_preds.cpu().numpy()
test_targets_np = test_targets.cpu().numpy()

# Concatenate predictions and targets
all_preds = np.concatenate([train_preds_np, val_preds_np, test_preds_np], axis=0)
all_targets = np.concatenate([train_targets_np, val_targets_np, test_targets_np], axis=0)

# Create labels
train_labels = np.array(['Train'] * len(train_preds_np))
val_labels = np.array(['Validation'] * len(val_preds_np))
test_labels = np.array(['Test'] * len(test_preds_np))
all_labels = np.concatenate([train_labels, val_labels, test_labels])

# Create a color map
colors = {'Train': 'blue', 'Validation': 'green', 'Test': 'red'}
color_list = [colors[label] for label in all_labels]

# Create parity plot
plt.figure(figsize=(8, 8))
plt.scatter(all_targets, all_preds, c=color_list, alpha=0.6)

# Plot y=x line
min_val = min(all_targets.min(), all_preds.min())
max_val = max(all_targets.max(), all_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

# Set limits with buffer
buffer = (max_val - min_val) * 0.05
plt.xlim([min_val - buffer, max_val + buffer])
plt.ylim([min_val - buffer, max_val + buffer])

# Labels and title
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Parity Plot')
plt.legend(handles=[
    plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
    plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
    plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
])
plt.show()

# first run :
# MSE: 0.006
# RMSE: 0.078
# SSE: 1278.019
# MAE: 0.067
# R2: 0.994
# MRE: 5204.053%
# Mean relative error is large, here is the median relative error:132.780%
# MDAPE: 132.780%


if False:
    ####### Generating predictions and targets for all datasets #########

    # Obtain predictions and targets for train, val, and test sets
    train_preds, train_targets = test_model_with_parity(model=model, test_data_loader=train, device=device,
                                            batch_size=batch_size, model_needs_frag=True)
    val_preds, val_targets = test_model_with_parity(model=model, test_data_loader=val, device=device,
                                        batch_size=batch_size, model_needs_frag=True)
    test_preds, test_targets = test_model_with_parity(model=model, test_data_loader=test, device=device,
                                        batch_size=batch_size, model_needs_frag=True)

    ####### Calculating Metrics #########

    train_mae = pred_metric(prediction=train_preds, target=train_targets, metrics='mae', print_out=False)['mae']
    val_mae = pred_metric(prediction=val_preds, target=val_targets, metrics='mae', print_out=False)['mae']
    test_mae = pred_metric(prediction=test_preds, target=test_targets, metrics='mae', print_out=False)['mae']

    overall_mae = (train_mae + val_mae + test_mae) / 3  # Assuming targets are not scaled
    print(f'Overall MAE {overall_mae:.3f}')

    ####### Creating Parity Plot #########

    # Convert tensors to numpy arrays
    train_preds_np = train_preds.cpu().detach().numpy()
    train_targets_np = train_targets.cpu().detach().numpy()
    val_preds_np = val_preds.cpu().detach().numpy()
    val_targets_np = val_targets.cpu().detach().numpy()
    test_preds_np = test_preds.cpu().detach().numpy()
    test_targets_np = test_targets.cpu().detach().numpy()

    # Concatenate predictions and targets
    all_preds = np.concatenate([train_preds_np, val_preds_np, test_preds_np])
    all_targets = np.concatenate([train_targets_np, val_targets_np, test_targets_np])

    # Create labels
    train_labels = np.array(['Train'] * len(train_preds_np))
    val_labels = np.array(['Validation'] * len(val_preds_np))
    test_labels = np.array(['Test'] * len(test_preds_np))
    all_labels = np.concatenate([train_labels, val_labels, test_labels])

    # Create a color map
    colors = {'Train': 'blue', 'Validation': 'green', 'Test': 'red'}
    color_list = [colors[label] for label in all_labels]

    # Create parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, c=color_list, alpha=0.6)

    # Plot y=x line
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    # Set limits with buffer
    buffer = (max_val - min_val) * 0.05
    plt.xlim([min_val - buffer, max_val + buffer])
    plt.ylim([min_val - buffer, max_val + buffer])

    # Labels and title
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Parity Plot')
    plt.legend(handles=[plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
                        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
                        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)])
    plt.show()