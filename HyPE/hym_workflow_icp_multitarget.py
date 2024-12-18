from grape_chem.models import GroupGAT_jittable
from grape_chem.utils import DataSet, train_model_jit, EarlyStopping, split_data, test_model_jit, pred_metric, return_hidden_layers, set_seed, JT_SubGraph
from grape_chem.datasets import FreeSolv
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Batch
from tqdm import tqdm
from typing import Union, List, Tuple
from torch import Tensor
import os

# Install GraPE with: pip install "git+https://github.com/aerte/GraPE.git#subdirectory=python"

def standardize(x, mean, std):
    return (x - mean) / std

##########################################################################################
#####################    Data Input Region  ##############################################
##########################################################################################

set_seed(42)

# Hyperparameters
epochs = 12000
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.001054627
weight_decay = 1e-4
mlp_layers = 2
atom_layers = 3
mol_layers = 3

# Change to your own specifications
root = './env/params_prediction.xlsx'
sheet_name = ''

df = pd.read_excel(root)
# Read SMILES and target properties A, B, C, D
smiles = df['SMILES'].to_numpy()
target_columns = ['A0', 'B0', 'C0', 'D0', 'E0']
targets = df[target_columns].to_numpy()  # Shape: (num_samples, 4)

# Read tags for custom splits
tags = df['bin'].to_numpy()
unique_tags = np.unique(tags)
#tag_to_int = {'Training': 0, 'Validation': 1, 'Test': 2}
tag_to_int = {'train': 0, 'val': 1, 'test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

# Global features
# global_feats = df['T'].to_numpy()

# Standardize targets separately
mean_targets = np.mean(targets, axis=0)  # Shape: (4,)
std_targets = np.std(targets, axis=0)
targets_standardized = (targets - mean_targets) / std_targets  # Shape: (num_samples, 4)

# Standardize global features
#mean_global_feats = np.mean(global_feats)
#std_global_feats = np.std(global_feats)
#global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

########################## Fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("Initializing fragmentation...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme)
frag_dim = fragmentation.frag_dim
print("Done.")

########################### DataSet Creation ######################################
data = DataSet(smiles=smiles, target=targets_standardized, global_features=None, filter=True, fragmentation=fragmentation, )

# Split data using custom splits
train, val, test = split_data(data, split_type='custom', custom_split=custom_split)

############################################################################################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define network parameters
mlp = return_hidden_layers(mlp_layers)
net_params = {
    "device": device,  # Shouldn't be passed in this way, but best we have for now
    "num_atom_type": 44,  # == node_in_dim TODO: check matches with featurizer or read from featurizer
    "num_bond_type": 12,  # == edge_in_dim
    "dropout": 0.0,
    "MLP_layers": mlp_layers,
    "frag_dim": frag_dim,
    "final_dropout": 0.119,
    "use_global_features": True,
    # For origins:
    "num_heads": 1,
    # For AFP:
    "node_in_dim": 44,
    "edge_in_dim": 12,
    "num_global_feats": 1,
    "hidden_dim": hidden_dim,  # Important: check matches with `L1_hidden_dim`
    "mlp_out_hidden": mlp,
    "num_layers_atom": atom_layers,
    "num_layers_mol": mol_layers,
    # For channels:
    "L1_layers_atom": 4,  # L1_layers
    "L1_layers_mol": 1,   # L1_depth
    "L1_dropout": 0.142,
    "L2_layers_atom": 2,  # L2_layers
    "L2_layers_mol": 3,   # L2_depth
    "L2_dropout": 0.255,
    "L3_layers_atom": 1,  # L3_layers
    "L3_layers_mol": 4,   # L3_depth
    "L3_dropout": 0.026,
    "L1_hidden_dim": 247,
    "L2_hidden_dim": 141,
    "L3_hidden_dim": 47,
    "output_dim": 4,      # Number of target properties
}

# Initialize the model with output dimension of 4
model = GroupGAT_jittable.GCGAT_v4pro_jit(net_params)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_Stopper = EarlyStopping(patience=50, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=1.00E-09, patience=15)

loss_func = torch.nn.functional.mse_loss

# Define model filename
model_filename = 'gcgat_jitted_latest.pth'

# Check if the model file exists
if os.path.exists(model_filename):
    print(f"Model file '{model_filename}' found. Loading the trained model.")
    # Load the model state dict
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
else:
    print(f"No trained model found at '{model_filename}'. Proceeding to train the model.")
    # Train the model
    train_model_jit(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_data_loader=train,
        val_data_loader=val,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
        scheduler=scheduler,
        model_needs_frag=True,
        net_params=net_params
    )
    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to '{model_filename}'.")

####### Generating prediction tensor for the TEST set (Not rescaled) #########
pred = test_model_jit(model=model, test_data_loader=test, device=device, batch_size=batch_size)

# Rescale predictions and targets
preds_rescaled = pred * std_targets + mean_targets
targets_rescaled = test.y * std_targets + mean_targets

# Calculate metrics for each property
for i, prop in enumerate(['A', 'B', 'C', 'D']):
    pred_prop = preds_rescaled[:, i]
    target_prop = targets_rescaled[:, i]
    mae = torch.mean(torch.abs(pred_prop - target_prop))
    print(f'MAE for property {prop}: {mae.item():.4f}')

####### Example for overall evaluation of the MAE #########
train_preds = test_model_jit(model=model, test_data_loader=train, device=device, batch_size=batch_size)
val_preds = test_model_jit(model=model, test_data_loader=val, device=device, batch_size=batch_size)

# Rescale predictions and targets
train_preds_rescaled = train_preds * std_targets + mean_targets
train_targets_rescaled = train.y * std_targets + mean_targets

val_preds_rescaled = val_preds * std_targets + mean_targets
val_targets_rescaled = val.y * std_targets + mean_targets

# Calculate overall MAE for each property
for i, prop in enumerate(['A', 'B', 'C', 'D']):
    train_mae = torch.mean(torch.abs(train_preds_rescaled[:, i] - train_targets_rescaled[:, i]))
    val_mae = torch.mean(torch.abs(val_preds_rescaled[:, i] - val_targets_rescaled[:, i]))
    test_mae = torch.mean(torch.abs(preds_rescaled[:, i] - targets_rescaled[:, i]))
    overall_mae = (train_mae + val_mae + test_mae) / 3
    print(f'Overall MAE for property {prop}: {overall_mae.item():.4f}')

# Modify the test_model function to return both predictions and targets
def test_model_with_parity(
    model: torch.nn.Module,
    test_data_loader: Union[list, DataLoader],
    device: str = None,
    batch_size: int = 32,
    return_latents: bool = False,
    model_needs_frag: bool = False
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """
    Auxiliary function to test a trained model and return the predictions as well as the targets and optional latent node
    representations. Can initialize DataLoaders if only list of Data objects are given.

    Parameters
    ------------
    model: torch.nn.Module
        Model that will be tested. Has to be a torch Module.
    test_data_loader: list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the test graphs.
    device: str
        Torch device to be used ('cpu', 'cuda' or 'mps'). Default: 'cpu'
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
        test_data_loader = DataLoader([data for data in test_data_loader], batch_size=batch_size)

    model.eval()

    with torch.no_grad():
        preds_list = []
        targets_list = []
        if return_latents:
            latents_list = []
        with tqdm(total=len(test_data_loader)) as pbar:
            for idx, batch in enumerate(test_data_loader):
                batch = batch.to(device)
                # Handling models that require fragment information
                if model_needs_frag:
                    if return_latents:
                        out, lat = model(batch, return_lats=True)
                        latents_list.append(lat.detach().cpu())
                    else:
                        out = model(batch)
                else:
                    if return_latents:
                        out, lat = model(batch, return_lats=True)
                        latents_list.append(lat.detach().cpu())
                    else:
                        out = model(batch)

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
train_preds, train_targets = test_model_with_parity(
    model=model,
    test_data_loader=train,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)

val_preds, val_targets = test_model_with_parity(
    model=model,
    test_data_loader=val,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)

test_preds, test_targets = test_model_with_parity(
    model=model,
    test_data_loader=test,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)

####### Rescale predictions and targets #########
train_preds_rescaled = train_preds * std_targets + mean_targets
train_targets_rescaled = train_targets * std_targets + mean_targets

val_preds_rescaled = val_preds * std_targets + mean_targets
val_targets_rescaled = val_targets * std_targets + mean_targets

test_preds_rescaled = test_preds * std_targets + mean_targets
test_targets_rescaled = test_targets * std_targets + mean_targets

####### Creating Parity Plot #########
import matplotlib.pyplot as plt

for i, prop in enumerate(['A', 'B', 'C', 'D']):
    # Combine all datasets
    all_preds = np.concatenate([
        train_preds_rescaled[:, i].cpu().numpy(),
        val_preds_rescaled[:, i].cpu().numpy(),
        test_preds_rescaled[:, i].cpu().numpy()
    ])
    all_targets = np.concatenate([
        train_targets_rescaled[:, i].cpu().numpy(),
        val_targets_rescaled[:, i].cpu().numpy(),
        test_targets_rescaled[:, i].cpu().numpy()
    ])

    # Create labels
    train_labels = np.array(['Train'] * len(train_preds_rescaled))
    val_labels = np.array(['Validation'] * len(val_preds_rescaled))
    test_labels = np.array(['Test'] * len(test_preds_rescaled))
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
    plt.title(f'Parity Plot for Property {prop}')
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
    ])
    plt.show()