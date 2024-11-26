###### with variable number of coeffs ###########

from grape_chem.models.AGC import AGC
from grape_chem.utils import (
    DataSet, train_model, EarlyStopping, split_data,
    test_model, pred_metric, return_hidden_layers,
    set_seed, JT_SubGraph
)
from grape_chem.datasets import FreeSolv
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Union, List, Tuple
from torch import Tensor
import os

# Install GraPE with: pip install "git+https://github.com/aerte/GraPE.git#subdirectory=python"

def standardize(x, mean, std):
    return (x - mean) / std

def split_data_by_indices(data, train_idx, val_idx, test_idx):
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]
    return train, val, test

def test_model_with_parity(
    model: torch.nn.Module,
    test_data_loader: Union[list, DataLoader],
    device: str = None,
    batch_size: int = 32,
    return_latents: bool = False,
    model_needs_frag: bool = False
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """
    Auxiliary function to test a trained model and return the predictions as well as the targets.
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

def train_and_evaluate_model(
    target_name, smiles, target_values, fragmentation,
    custom_split_indices, net_params, device, batch_size, epochs,
    learning_rate, weight_decay, scheduler_patience, patience,
    global_feats = None
):
    # Standardize target
    mean_target = np.mean(target_values)
    std_target = np.std(target_values)
    target_standardized = standardize(target_values, mean_target, std_target)

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

    # Initialize model
    model = AGC(net_params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, min_lr=1.00E-09, patience=scheduler_patience
    )
    early_Stopper = EarlyStopping(patience=patience, model_name='random', skip_save=True)
    loss_func = torch.nn.functional.mse_loss

    # Define model filename
    model_filename = f'AGC_dst_model_subpart_{target_name}.pth'

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
    test_pred_rescaled= test_pred_rescaled.to(device)

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
    plt.title(f'Parity Plot for Property {target_name} for decoupled single-task (AGC)')
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
    ])
    plt.show()

##########################################################################################
############################### New Function #############################################
##########################################################################################

def generate_predictions_and_save(
    results_all: dict,
    df: pd.DataFrame,
    smiles: np.ndarray,
    batch_size: int,
    device: torch.device,
    output_filename: str
):
    """
    Function to generate predictions for all targets, align them with the SMILES, and save to a CSV file.
    """
    from tqdm import tqdm
    from rdkit import Chem

    # Step 1: Get valid indices using RDKit
    def get_valid_indices(smiles_list):
        valid_indices = []
        for idx, smile in enumerate(tqdm(smiles_list, desc='Validating SMILES')):
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                valid_indices.append(idx)
        return valid_indices

    print("Identifying valid SMILES...")
    valid_indices = get_valid_indices(smiles)
    print(f"Number of valid SMILES: {len(valid_indices)} out of {len(smiles)}")

    # Step 2: Filter df and smiles
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    smiles_filtered = smiles[valid_indices]

    # Step 3: Initialize DataFrame for predictions
    df_pred = df_filtered.copy()

    # Step 4: Generate predictions for each target
    for target_name, results in results_all.items():
        print(f"Generating predictions for target: {target_name}")
        model = results['model']
        mean = results['mean']
        std = results['std']

        # Create DataSet for prediction
        data = DataSet(
            smiles=smiles_filtered,
            target=np.zeros(len(smiles_filtered)),  # Dummy target
            global_features=None,
            filter=True,
            fragmentation=fragmentation  # Assuming 'fragmentation' is accessible here
        )

        # Generate predictions
        preds = test_model(
            model=model,
            test_data_loader=data,
            device=device,
            batch_size=batch_size,
        )

        # Rescale predictions
        preds_rescaled = preds * std + mean

        # Add predictions to DataFrame
        df_pred[f'Predicted_{target_name}'] = preds_rescaled.detach().cpu().numpy()

    # Step 5: Save to CSV
    df_pred.to_csv(output_filename, index=False)
    print(f"All predictions saved to '{output_filename}'.")

##########################################################################################
############################### Main Code ################################################
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
scheduler_patience = 15  # For learning rate scheduler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Change to your own specifications
root = './env/params_prediction.xlsx'
sheet_name = ''

df = pd.read_excel(root)
# Read SMILES and target properties
smiles = df['SMILES'].to_numpy()
targets_df = df[['A0', 'B0', 'C0', 'D0', 'E0']]  # You can have any number of targets here
target_names = targets_df.columns.tolist()
targets = targets_df.to_numpy()  # Shape: (num_samples, num_targets)

# Read tags for custom splits
tags = df['bin'].to_numpy()
unique_tags = np.unique(tags)
tag_to_int = {'train': 0, 'val': 1, 'test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

# Get indices for splits
train_indices = np.where(custom_split == 0)[0]
val_indices = np.where(custom_split == 1)[0]
test_indices = np.where(custom_split == 2)[0]
custom_split_indices = (train_indices, val_indices, test_indices)

# Global features
#global_feats = df['T'].to_numpy()
# Standardize global features
# mean_global_feats = np.mean(global_feats)
# std_global_feats = np.std(global_feats)
# global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

########################## Fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("Initializing fragmentation...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path="env/decoupled_single_task_frag.pth")
frag_dim = fragmentation.frag_dim
print("Done.")

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
    "use_global_features": False,
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

# Dictionary to store results for each target
results_all = {}

# Loop over each target
for idx, target_name in enumerate(target_names):
    print(f"\nProcessing target: {target_name}")
    target_values = targets[:, idx]
    results = train_and_evaluate_model(
        target_name=target_name,
        smiles=smiles,
        target_values=target_values,
        fragmentation=fragmentation,
        custom_split_indices=custom_split_indices,
        net_params=net_params,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_patience=scheduler_patience,
        patience=patience  # Ensure 'patience' is defined
    )
    results_all[target_name] = results

# Create parity plots
for target_name in target_names:
    print(f"\nCreating parity plot for target: {target_name}")
    results = results_all[target_name]
    create_parity_plot(results, target_name)

##########################################################################################
############################### Generate and Save Predictions ###########################
##########################################################################################

# Define output filename
output_filename = 'predictions_AGC_decoupled_single-task.csv'

# Generate predictions and save to CSV
generate_predictions_and_save(
    results_all=results_all,
    df=df,
    smiles=smiles,
    batch_size=batch_size,
    device=device,
    output_filename=output_filename
)
