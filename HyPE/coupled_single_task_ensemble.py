import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os

from grape_chem.models import GroupGAT_Ensemble
from grape_chem.utils import (
    DataSet, train_model, EarlyStopping, split_data,
    test_model, pred_metric, return_hidden_layers,
    set_seed, JT_SubGraph
)
from torch.optim import lr_scheduler

def standardize(x, mean, std):
    return (x - mean) / std

def split_data_by_indices(data, train_idx, val_idx, test_idx):
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]
    return train, val, test


def train_and_evaluate_model(
    target_name, smiles, target_values, fragmentation,
    custom_split_indices, net_params, device, batch_size, epochs,
    learning_rate, weight_decay, scheduler_patience, global_feats=None
):
    # Standardize target
    mean_target = np.mean(target_values)
    std_target = np.std(target_values)
    target_standardized = standardize(target_values, mean_target, std_target)
    if global_feats is not None:
        global_feats = standardize(global_feats, np.mean(global_feats), np.std(global_feats))
    # Create DataSet
    data = DataSet(
        smiles=smiles,
        target=target_standardized,
        global_features=global_feats,
        filter=True,
        fragmentation=fragmentation
    )

    # Split data
    train_indices, val_indices, test_indices = custom_split_indices
    train_data, val_data, test_data = split_data_by_indices(
        data, train_indices, val_indices, test_indices
    )

    # Initialize the ensemble model
    num_models = 5  # Number of models in the ensemble
    model = GroupGAT_Ensemble(net_params, num_models).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, min_lr=1.00E-09, patience=scheduler_patience
    )
    early_Stopper = EarlyStopping(patience=patience, model_name='random', skip_save=True)
    loss_func = torch.nn.functional.mse_loss

    # Define model filename
    model_filename = f'gcgat_ensemble_model_{target_name}.pth'

    # Check if the model file exists
    if os.path.exists(model_filename):
        print(f"Model file '{model_filename}' found. Loading the trained model.")
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.eval()
    else:
        print(f"No trained model found at '{model_filename}'. Proceeding to train the model.")
        # Train the model
        train_model(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            train_data_loader=train_data,
            val_data_loader=val_data,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            scheduler=scheduler,
            model_needs_frag=True,
            early_stopper=early_Stopper
        )
        # Save the trained model
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to '{model_filename}'.")

    ####### Evaluating the Model #########
    # Generate predictions on test set
    test_pred = test_model(
        model=model,
        test_data_loader=test_data,
        device=device,
        batch_size=batch_size
    )
    test_target = torch.tensor([data.y for data in test_data]).to(device)

    # Rescale predictions and targets
    test_pred_rescaled = test_pred * std_target + mean_target
    test_target_rescaled = test_target * std_target + mean_target

    # Calculate metrics
    mae = torch.mean(torch.abs(test_pred_rescaled - test_target_rescaled))
    print(f'MAE for property {target_name} on test set: {mae.item():.4f}')

    # Generate predictions on train and val sets
    train_pred = test_model(
        model=model,
        test_data_loader=train_data,
        device=device,
        batch_size=batch_size
    )
    val_pred = test_model(
        model=model,
        test_data_loader=val_data,
        device=device,
        batch_size=batch_size
    )
    train_target = torch.tensor([data.y for data in train_data]).to(device)
    val_target = torch.tensor([data.y for data in val_data]).to(device)

    # Rescale predictions and targets
    train_pred_rescaled = train_pred * std_target + mean_target
    val_pred_rescaled = val_pred * std_target + mean_target
    train_target_rescaled = train_target * std_target + mean_target
    val_target_rescaled = val_target * std_target + mean_target

    # Store results
    results = {
        'model': model,
        'mean': mean_target,
        'std': std_target,
        'train_pred': train_pred_rescaled.detach().cpu().numpy(),
        'train_target': train_target_rescaled.detach().cpu().numpy(),
        'val_pred': val_pred_rescaled.detach().cpu().numpy(),
        'val_target': val_target_rescaled.detach().cpu().numpy(),
        'test_pred': test_pred_rescaled.detach().cpu().numpy(),
        'test_target': test_target_rescaled.detach().cpu().numpy(),
    }

    return results


##########################################################################################
############################### Main Code ################################################
##########################################################################################

# Set seed for reproducibility
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
scheduler_patience = 15  # For learning rate scheduler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Change to your own specifications
root = './env/ICP.xlsx'
sheet_name = ''

df = pd.read_excel(root)
# Read SMILES and target properties A, B, C, D
smiles = df['SMILES'].to_numpy()
targets = df['Value'].to_numpy()

# Read tags for custom splits
tags = df['Subset'].to_numpy()
unique_tags = np.unique(tags)
tag_to_int = {'Training': 0, 'Validation': 1, 'Test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

# Global features
global_feats = df['T'].to_numpy()

# Get indices for splits
train_indices = np.where(custom_split == 0)[0]
val_indices = np.where(custom_split == 1)[0]
test_indices = np.where(custom_split == 2)[0]
custom_split_indices = (train_indices, val_indices, test_indices)

# Global features (if any)
# global_feats = df['T'].to_numpy()
# Standardize global features if using
# mean_global_feats = np.mean(global_feats)
# std_global_feats = np.std(global_feats)
# global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

########################## Fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("Initializing fragmentation...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path='env/ICP_fragmentation.pth')
frag_dim = fragmentation.frag_dim
print("Done.")

###################### train and eval model 



############################## workflow ##########################################
# Define network parameters
mlp = return_hidden_layers(mlp_layers)
net_params = {
    "device": device,  # Shouldn't be passed in this way, but best we have for now
    "num_atom_type": 44,  # Should match your featurizer's output
    "num_bond_type": 12,  # Should match your featurizer's output
    "dropout": 0.0,
    "MLP_layers": mlp_layers,
    "frag_dim": frag_dim,
    "final_dropout": 0.119,
    "use_global_features": False,
    "global_features": 1,
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

# Initialize the ensemble model
num_models = 5  # Number of models in the ensemble
# Ensure each model has its own parameters if needed
model = GroupGAT_Ensemble(net_params, num_models).to(device)

# Training the ensemble model
results = train_and_evaluate_model(
    target_name="ICP",
    smiles=smiles,
    target_values=targets,
    fragmentation=fragmentation,
    custom_split_indices=custom_split_indices,
    net_params=net_params,
    device=device,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    scheduler_patience=scheduler_patience,
    global_feats=global_feats
)

# Create parity plot for the overall predictions
def create_parity_plot(results_dict, target_name):
    train_preds_np = results_dict['train_pred']
    val_preds_np = results_dict['val_pred']
    test_preds_np = results_dict['test_pred']

    train_targets_np = results_dict['train_target']
    val_targets_np = results_dict['val_target']
    test_targets_np = results_dict['test_target']

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
    plt.title(f'Parity Plot for Property {target_name}')
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
    ])
    plt.show()

# Create parity plot
print(f"\nCreating parity plot for target: {target_name}")
create_parity_plot(results, target_name)

