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
targets = df[['A0', 'B0', 'C0', 'D0', 'E0']].to_numpy()  # Shape: (num_samples, 4)

# Read tags for custom splits
tags = df['Subset'].to_numpy()
unique_tags = np.unique(tags)
tag_to_int = {'train': 0, 'val': 1, 'test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

# Global features
global_feats = df['T'].to_numpy()

# Standardize global features
mean_global_feats = np.mean(global_feats)
std_global_feats = np.std(global_feats)
global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

########################## Fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("Initializing fragmentation...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme)
frag_dim = fragmentation.frag_dim
print("Done.")

########################### DataSet Creation ######################################
# Separate targets
target_A = targets[:, 0]
target_B = targets[:, 1]
target_C = targets[:, 2]
target_D = targets[:, 3]

# Standardize each target separately
mean_A, std_A = np.mean(target_A), np.std(target_A)
mean_B, std_B = np.mean(target_B), np.std(target_B)
mean_C, std_C = np.mean(target_C), np.std(target_C)
mean_D, std_D = np.mean(target_D), np.std(target_D)

target_A_standardized = standardize(target_A, mean_A, std_A)
target_B_standardized = standardize(target_B, mean_B, std_B)
target_C_standardized = standardize(target_C, mean_C, std_C)
target_D_standardized = standardize(target_D, mean_D, std_D)

# Create separate DataSets
data_A = DataSet(smiles=smiles, target=target_A_standardized, global_features=global_feats, filter=True, fragmentation=fragmentation)
data_B = DataSet(smiles=smiles, target=target_B_standardized, global_features=global_feats, filter=True, fragmentation=fragmentation)
data_C = DataSet(smiles=smiles, target=target_C_standardized, global_features=global_feats, filter=True, fragmentation=fragmentation)
data_D = DataSet(smiles=smiles, target=target_D_standardized, global_features=global_feats, filter=True, fragmentation=fragmentation)

# Split indices for train, val, test
train_indices = np.where(custom_split == 0)[0]
val_indices = np.where(custom_split == 1)[0]
test_indices = np.where(custom_split == 2)[0]

# Function to split data by indices
def split_data_by_indices(data, train_idx, val_idx, test_idx):
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]
    return train, val, test

# Split datasets
train_A, val_A, test_A = split_data_by_indices(data_A, train_indices, val_indices, test_indices)
train_B, val_B, test_B = split_data_by_indices(data_B, train_indices, val_indices, test_indices)
train_C, val_C, test_C = split_data_by_indices(data_C, train_indices, val_indices, test_indices)
train_D, val_D, test_D = split_data_by_indices(data_D, train_indices, val_indices, test_indices)

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
    "output_dim": 1,      # Output dimension is 1 for each model
}

# Initialize models for each property
model_A = GroupGAT_jittable.GCGAT_v4pro_jit(net_params).to(device)
model_B = GroupGAT_jittable.GCGAT_v4pro_jit(net_params).to(device)
model_C = GroupGAT_jittable.GCGAT_v4pro_jit(net_params).to(device)
model_D = GroupGAT_jittable.GCGAT_v4pro_jit(net_params).to(device)

# Optimizers
optimizer_A = torch.optim.Adam(model_A.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_B = torch.optim.Adam(model_B.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_C = torch.optim.Adam(model_C.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Schedulers
scheduler_A = lr_scheduler.ReduceLROnPlateau(optimizer_A, mode='min', factor=0.7, min_lr=1.00E-09, patience=15)
scheduler_B = lr_scheduler.ReduceLROnPlateau(optimizer_B, mode='min', factor=0.7, min_lr=1.00E-09, patience=15)
scheduler_C = lr_scheduler.ReduceLROnPlateau(optimizer_C, mode='min', factor=0.7, min_lr=1.00E-09, patience=15)
scheduler_D = lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.7, min_lr=1.00E-09, patience=15)

loss_func = torch.nn.functional.mse_loss

# Early Stoppers (optional)
early_Stopper_A = EarlyStopping(patience=50, model_name='model_A', skip_save=True)
early_Stopper_B = EarlyStopping(patience=50, model_name='model_B', skip_save=True)
early_Stopper_C = EarlyStopping(patience=50, model_name='model_C', skip_save=True)
early_Stopper_D = EarlyStopping(patience=50, model_name='model_D', skip_save=True)

# Dictionaries to organize models and data
models = {'A': model_A, 'B': model_B, 'C': model_C, 'D': model_D}
optimizers = {'A': optimizer_A, 'B': optimizer_B, 'C': optimizer_C, 'D': optimizer_D}
schedulers = {'A': scheduler_A, 'B': scheduler_B, 'C': scheduler_C, 'D': scheduler_D}
train_loaders = {'A': train_A, 'B': train_B, 'C': train_C, 'D': train_D}
val_loaders = {'A': val_A, 'B': val_B, 'C': val_C, 'D': val_D}
test_loaders = {'A': test_A, 'B': test_B, 'C': test_C, 'D': test_D}
means = {'A': mean_A, 'B': mean_B, 'C': mean_C, 'D': mean_D}
stds = {'A': std_A, 'B': std_B, 'C': std_C, 'D': std_D}

# Training loop for each property
for prop in ['A', 'B', 'C', 'D']:
    print(f"\nTraining model for property {prop}")
    model = models[prop]
    optimizer = optimizers[prop]
    scheduler = schedulers[prop]
    train_loader = train_loaders[prop]
    val_loader = val_loaders[prop]
    test_loader = test_loaders[prop]
    mean_target = means[prop]
    std_target = stds[prop]
    
    # Define model filename
    model_filename = f'gcgat_model_{prop}.pth'

    # Check if the model file exists
    if os.path.exists(model_filename):
        print(f"Model file '{model_filename}' found. Loading the trained model.")
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.eval()
    else:
        print(f"No trained model found at '{model_filename}'. Proceeding to train the model.")
        # Train the model
        train_model_jit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            train_data_loader=train_loader,
            val_data_loader=val_loader,
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

    ####### Evaluating the Model #########
    # Generate predictions on test set
    pred = test_model_jit(model=model, test_data_loader=test_loader, device=device, batch_size=batch_size)
    target = torch.tensor([data.y for data in test_loader]).to(device)

    # Rescale predictions and targets
    pred_rescaled = pred * std_target + mean_target
    target_rescaled = target * std_target + mean_target

    # Calculate metrics
    mae = torch.mean(torch.abs(pred_rescaled - target_rescaled))
    print(f'MAE for property {prop} on test set: {mae.item():.4f}')

    # Store predictions and targets for parity plot
    models[prop] = model  # Update the model after training
    models[f'pred_{prop}'] = pred_rescaled.detach().cpu().numpy()
    models[f'target_{prop}'] = target_rescaled.detach().cpu().numpy()

####### Creating Parity Plots #########
import matplotlib.pyplot as plt

for prop in ['A', 'B', 'C', 'D']:
    model = models[prop]
    mean_target = means[prop]
    std_target = stds[prop]
    train_loader = train_loaders[prop]
    val_loader = val_loaders[prop]
    test_loader = test_loaders[prop]

    # Generate predictions on train, val, test sets
    train_pred = test_model_jit(model=model, test_data_loader=train_loader, device=device, batch_size=batch_size)
    val_pred = test_model_jit(model=model, test_data_loader=val_loader, device=device, batch_size=batch_size)
    test_pred = models[f'pred_{prop}']  # Already computed

    # Collect targets
    train_target = torch.tensor([data.y for data in train_loader]).to(device)
    val_target = torch.tensor([data.y for data in val_loader]).to(device)
    test_target = models[f'target_{prop}']  # Already computed

    # Rescale predictions and targets
    train_pred_rescaled = train_pred * std_target + mean_target
    val_pred_rescaled = val_pred * std_target + mean_target
    # test_pred_rescaled and test_target_rescaled are already rescaled

    train_target_rescaled = train_target * std_target + mean_target
    val_target_rescaled = val_target * std_target + mean_target

    # Convert to numpy arrays
    train_preds_np = train_pred_rescaled.detach().cpu().numpy()
    val_preds_np = val_pred_rescaled.detach().cpu().numpy()
    test_preds_np = test_pred

    train_targets_np = train_target_rescaled.detach().cpu().numpy()
    val_targets_np = val_target_rescaled.detach().cpu().numpy()
    test_targets_np = test_target

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
    plt.title(f'Parity Plot for Property {prop}')
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
    ])
    plt.show()


