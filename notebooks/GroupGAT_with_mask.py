from grape_chem.models import GroupGAT_jittable
from grape_chem.utils import (
    DataSet,
    train_model_jit,
    EarlyStopping,
    split_data,
    test_model_jit,
    pred_metric,
    return_hidden_layers,
    set_seed,
    JT_SubGraph,
)
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Batch, Data
from tqdm import tqdm
from typing import Union, List, Tuple, Callable
from torch import Tensor
import torch.nn.functional as F
import os

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
learning_rate = 0.001054627
weight_decay = 1e-4
mlp_layers = 2
atom_layers = 3
mol_layers = 3

# Change to your own specifications
root = './env/critprop_expt_data_train_random_fold0.csv'
sheet_name = ''

df = pd.read_csv(root)
# Read SMILES and target properties
smiles = df['smiles'].to_numpy()
target_columns = ["Tc","Pc","rhoc","omega","Tb","Tm","dHvap","dHfus"]
targets_df = df[target_columns]  # You can have any number of targets here
num_targets = targets_df.shape[1]
target_names = targets_df.columns.tolist()
targets = targets_df.to_numpy()  # Shape: (num_samples, num_targets)

# Read tags for custom splits
tags = df['bin'].to_numpy()
unique_tags = np.unique(tags)
tag_to_int = {'train': 0, 'val': 1, 'test': 2}
#custom_split = np.array([tag_to_int[tag] for tag in tags])

# Check if there are any missing targets
if np.isnan(targets).any():
    # Create mask: True where targets are present
    mask = ~np.isnan(targets)  # Boolean array where True indicates presence
    mask = mask.astype(np.float32)  # Convert to float32 for tensor operations
    mask = None
    # Fill missing targets with zeros (since they'll be masked out during loss computation)
    targets_filled = np.nan_to_num(targets, nan=0.0)

    # Compute mean and std excluding NaNs
    mean_targets = np.nanmean(targets, axis=0)  # Shape: (num_targets,)
    std_targets = np.nanstd(targets, axis=0)

    # Standardize targets
    targets_standardized = standardize(targets, mean_targets, std_targets)
    # Replace NaNs with zeros
    targets_standardized_filled = np.nan_to_num(targets_standardized, nan=0.0)
else:
    mask = None  # No mask needed when all targets are present
    targets_filled = targets

    # Compute mean and std normally
    mean_targets = np.mean(targets, axis=0)
    std_targets = np.std(targets, axis=0)

    # Standardize targets
    targets_standardized = standardize(targets, mean_targets, std_targets)
    targets_standardized_filled = targets_standardized

########################## Fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("Initializing fragmentation...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path="env/multitask_groupGAT_frag_critprop.pth")
frag_dim = fragmentation.frag_dim
print("Done.")

########################### DataSet Creation ######################################

# Include 'mask' in DataSet creation if mask is not None
data = DataSet(
    smiles=smiles,
    target=targets_standardized_filled,
    mask=mask,  # This can be None if no mask is needed
    filter=True,
    fragmentation=fragmentation,
    target_columns=target_columns,
)

# Split data using custom splits
train, val, test = split_data(data, split_type='consecutive', split_frac=(0.8, 0.1, 0.1))

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
    "output_dim": num_targets,  # Number of target properties
}

# Initialize the model with output dimension equal to number of targets
model = GroupGAT_jittable.GCGAT_v4pro_jit(net_params)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_Stopper = EarlyStopping(patience=50, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, min_lr=1.00E-09, patience=15
)

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
        net_params=net_params,
        early_stopper=early_Stopper,
    )
    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to '{model_filename}'.")

####### Generating prediction tensor for the TEST set (Not rescaled) #########
pred = test_model_jit(model=model, test_data_loader=test, device=device, batch_size=batch_size)

# Rescale predictions and targets
preds_rescaled = pred * std_targets + mean_targets
targets_rescaled = test.y * std_targets + mean_targets

# Function to calculate and print MAE for each property
def calculate_mae(preds_rescaled, targets_rescaled, mask=None, dataset_name='Test'):
    maes = []
    for i, prop in enumerate(target_columns):
        pred_prop = preds_rescaled[:, i]
        target_prop = targets_rescaled[:, i]
        if mask is not None:
            mask_prop = mask[:, i]
            # Filter out missing targets
            pred_prop = pred_prop[mask_prop == 1]
            target_prop = target_prop[mask_prop == 1]
        mae = torch.mean(torch.abs(pred_prop - target_prop))
        print(f'MAE for property {prop} on {dataset_name} set: {mae.item():.4f}')
        maes.append(mae.item())
    if mask is not None:
        overall_mae = torch.mean(torch.abs(preds_rescaled[mask == 1] - targets_rescaled[mask == 1]))
    else:
        overall_mae = torch.mean(torch.abs(preds_rescaled - targets_rescaled))
    print(f'Overall MAE across all properties on {dataset_name} set: {overall_mae.item():.4f}')
    return maes, overall_mae.item()

def compute_metrics(preds_rescaled, targets_rescaled, mask=None, dataset_name='Test'):
    metrics_per_property = []
    for i, prop in enumerate(target_columns):
        pred_prop = preds_rescaled[:, i]
        target_prop = targets_rescaled[:, i]
        if mask is not None:
            mask_prop = mask[:, i]
            # Filter out missing targets
            pred_prop = pred_prop[mask_prop == 1]
            target_prop = target_prop[mask_prop == 1]
        # Call pred_metric to compute desired metrics
        results = pred_metric(pred_prop, target_prop, metrics='all', print_out=False)
        print(f"Metrics for property {prop} on {dataset_name} set:")
        for metric_name, value in results.items():
            print(f"{metric_name.upper()}: {value:.4f}")
        metrics_per_property.append(results)
    # Compute overall metrics
    if mask is not None:
        overall_mask = mask == 1
        overall_pred = preds_rescaled[overall_mask]
        overall_target = targets_rescaled[overall_mask]
    else:
        overall_pred = preds_rescaled.view(-1)
        overall_target = targets_rescaled.view(-1)
    overall_results = pred_metric(overall_pred, overall_target, metrics='all', print_out=False)
    print(f"Overall metrics across all properties on {dataset_name} set:")
    for metric_name, value in overall_results.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    return metrics_per_property, overall_results

# Calculate metrics for the test set
test_maes, test_overall_mae = calculate_mae(preds_rescaled, targets_rescaled, getattr(test, 'mask', None), dataset_name='Test')

####### Example for overall evaluation of the MAE #########
train_preds = test_model_jit(model=model, test_data_loader=train, device=device, batch_size=batch_size)
val_preds = test_model_jit(model=model, test_data_loader=val, device=device, batch_size=batch_size)

# Rescale predictions and targets
train_preds_rescaled = train_preds * std_targets + mean_targets
train_targets_rescaled = train.y * std_targets + mean_targets

val_preds_rescaled = val_preds * std_targets + mean_targets
val_targets_rescaled = val.y * std_targets + mean_targets

# Calculate metrics for train and validation sets
train_maes, train_overall_mae = calculate_mae(train_preds_rescaled, train_targets_rescaled, getattr(train, 'mask', None), dataset_name='Train')
val_maes, val_overall_mae = calculate_mae(val_preds_rescaled, val_targets_rescaled, getattr(val, 'mask', None), dataset_name='Validation')

# Adjust overall MAE across all datasets
overall_mae_per_property = []
for i in range(num_targets):
    overall_mae = (train_maes[i] + val_maes[i] + test_maes[i]) / 3
    overall_prop = target_columns[i]
    print(f'Overall MAE for property {overall_prop}: {overall_mae:.4f}')
    overall_mae_per_property.append(overall_mae)

# Overall MAE across all properties and datasets
overall_mae_all = (train_overall_mae + val_overall_mae + test_overall_mae) / 3
print(f'Overall MAE across all properties and datasets: {overall_mae_all:.4f}')
