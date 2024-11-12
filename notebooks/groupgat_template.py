import os
import copy
import yaml
import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader

from grape_chem.utils import (
    DataSet,
    split_data,
    train_model_jit,
    EarlyStopping,
    test_model_jit,
    pred_metric,
    return_hidden_layers,
    set_seed,
    JT_SubGraph,
)
from grape_chem.models import GroupGAT_jittable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Batch, Data
from tqdm import tqdm

# Function to standardize data
def standardize(x, mean, std):
    return (x - mean) / std

# Set seed for reproducibility
set_seed(42)

##########################################################################################
#####################    Data Input and Preprocessing  ###################################
##########################################################################################

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Extract configurations for initial dataset
file_name = data_config.get('file_name')
file_type = data_config.get('file_type', 'csv')
smiles_column = data_config.get('smiles_column', 'SMILES')
target_columns = data_config.get('target_columns', ['Target'])
global_features_columns = data_config.get('global_features_columns', None)
split_column = data_config.get('split_column', None)
sheet_name = data_config.get('sheet_name', 0)
fragmentation_settings = data_config.get('fragmentation', None)

# Load initial dataset
if file_type == 'csv':
    df = pd.read_csv(file_name)
elif file_type == 'excel':
    df = pd.read_excel(file_name, sheet_name=sheet_name)
else:
    raise ValueError(f"Unsupported file type: {file_type}")

# Read SMILES and targets
smiles = df[smiles_column].to_numpy()

targets_df = df[target_columns]
num_targets = targets_df.shape[1]
target_names = targets_df.columns.tolist()
targets = targets_df.to_numpy()  # Shape: (num_samples, num_targets)

# Read global features if provided
if global_features_columns is not None:
    global_feats = df[global_features_columns].to_numpy()
else:
    global_feats = None

# Handle missing targets
# TODO: refactor this to a function in utils
if np.isnan(targets).any():
    # Create mask: True where targets are present
    mask = ~np.isnan(targets)  # Boolean array where True indicates presence
    mask = mask.astype(np.float32)  # Convert to float32 for tensor operations
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

# Handle fragmentation
if fragmentation_settings is not None:
    # Get fragmentation settings
    scheme = fragmentation_settings.get('scheme', 'MG_plus_reference')
    save_file_path = fragmentation_settings.get('frag_save_file_path', None)
    tl_save_file_path = fragmentation_settings.get('transfer_learning_frag_save_file_path', None)
    verbose = fragmentation_settings.get('verbose', False)
    print("Initializing fragmentation...")
    fragmentation = JT_SubGraph(scheme=scheme, save_file_path=save_file_path, verbose=verbose)
    fragmentation_tl = JT_SubGraph(scheme=scheme, save_file_path=tl_save_file_path, verbose=verbose)
    frag_dim = fragmentation.frag_dim
    print("Done.")
else:
    fragmentation = None
    frag_dim = 0

# Handle custom splits
if split_column is not None:
    split_values = df[split_column].to_numpy()
    # Map split values to integers
    unique_splits = np.unique(split_values)
    split_mapping = {split: idx for idx, split in enumerate(unique_splits)}
    custom_split = np.array([split_mapping[split] for split in split_values])
else:
    # If no split column provided, split using ratio
    split_ratios = data_config.get('split_ratios', [0.8, 0.1, 0.1])
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(smiles))
    train_idx, test_idx = train_test_split(indices, test_size=split_ratios[2], random_state=42)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]), random_state=42
    )
    custom_split = np.zeros(len(smiles), dtype=int)
    custom_split[val_idx] = 1
    custom_split[test_idx] = 2

# Create Dataset
data = DataSet(
    smiles=smiles,
    target=targets_standardized_filled,
    global_features=global_feats,
    mask=mask,  # This can be None if no mask is needed
    filter=True,
    fragmentation=fragmentation,
    target_columns=target_columns,
    custom_split=custom_split,
)

# Split data
train_set, val_set, test_set = split_data(data, split_type='custom', custom_split=data.custom_split)

# Create DataLoaders
batch_size = data_config.get('batch_size', 5000)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

##########################################################################################
#####################    Model Setup and Training  #######################################
##########################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define network parameters
# Here we extract model parameters from the config file
# if none are provided we use the defaults hardcoded here

model_config = data_config.get('model', {})
mlp_layers = model_config.get('mlp_layers', 2)
mlp = return_hidden_layers(mlp_layers)

net_params = {
    "device": device,
    "num_atom_type": model_config.get('num_atom_type', 44),
    "num_bond_type": model_config.get('num_bond_type', 12),
    "dropout": model_config.get('dropout', 0.0),
    "MLP_layers": mlp_layers,
    "frag_dim": frag_dim,
    "final_dropout": model_config.get('final_dropout', 0.119),
    "use_global_features": model_config.get('use_global_features', True),
    "num_heads": model_config.get('num_heads', 1),
    "node_in_dim": model_config.get('node_in_dim', 44),
    "edge_in_dim": model_config.get('edge_in_dim', 12),
    "num_global_feats": global_feats.shape[1] if global_feats is not None else 0,
    "hidden_dim": model_config.get('hidden_dim', 47),
    "mlp_out_hidden": mlp,
    "num_layers_atom": model_config.get('num_layers_atom', 3),
    "num_layers_mol": model_config.get('num_layers_mol', 3),
    "L1_layers_atom": model_config.get('L1_layers_atom', 4),
    "L1_layers_mol": model_config.get('L1_layers_mol', 1),
    "L1_dropout": model_config.get('L1_dropout', 0.142),
    "L2_layers_atom": model_config.get('L2_layers_atom', 2),
    "L2_layers_mol": model_config.get('L2_layers_mol', 3),
    "L2_dropout": model_config.get('L2_dropout', 0.255),
    "L3_layers_atom": model_config.get('L3_layers_atom', 1),
    "L3_layers_mol": model_config.get('L3_layers_mol', 4),
    "L3_dropout": model_config.get('L3_dropout', 0.026),
    "L1_hidden_dim": model_config.get('L1_hidden_dim', 247),
    "L2_hidden_dim": model_config.get('L2_hidden_dim', 141),
    "L3_hidden_dim": model_config.get('L3_hidden_dim', 47),
    "output_dim": num_targets,  # Number of target properties
}

# Initialize model
model = GroupGAT_jittable.GCGAT_v4pro_jit(net_params)
model.to(device)

epochs = data_config.get('epochs', 1000)
learning_rate = data_config.get('learning_rate', 0.001054627)
weight_decay = data_config.get('weight_decay', 1e-4)
patience = data_config.get('patience', 30)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, min_lr=1.00E-09, patience=15
)
early_stopper = EarlyStopping(patience=50, model_name='random', skip_save=True)
loss_func = torch.nn.functional.mse_loss

model_filename = data_config.get('model_save_path', 'gcgat_jitted_model.pth')

# Load model if saved
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
        net_params=net_params,
        early_stopper=early_stopper,
    )
    # Save trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to '{model_filename}'.")

##########################################################################################
#####################    Evaluation and Metrics  #########################################
##########################################################################################

# Function to calculate and print metrics
def compute_metrics(preds_rescaled, targets_rescaled, mask=None, dataset_name='Test'):
    metrics_per_property = []
    num_targets = len(target_columns)
    # Ensure preds_rescaled and targets_rescaled are at least 2D
    if preds_rescaled.dim() == 1:
        preds_rescaled = preds_rescaled.unsqueeze(1)
        targets_rescaled = targets_rescaled.unsqueeze(1)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(1)
    for i, prop in enumerate(target_columns):
        pred_prop = preds_rescaled[:, i]
        target_prop = targets_rescaled # CURRENTLY HARDCODED TO 1-D targets !!
        if mask is not None:
            mask_prop = mask[:, i]
            # Filter out missing targets
            pred_prop = pred_prop[mask_prop == 1]
            target_prop = target_prop[mask_prop == 1]
        # Call pred_metric to compute desired metrics
        results = pred_metric(pred_prop, target_prop, metrics=['mae', 'r2'], print_out=False)
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

# Evaluate on test set
test_preds = test_model_jit(
    model=model,
    test_data_loader=test_loader,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True,
    net_params=net_params,
)

# Rescale predictions and targets
test_preds_rescaled = test_preds * std_targets + mean_targets
test_targets_rescaled = torch.cat([data.y for data in test_loader], dim=0) * std_targets + mean_targets

# Ensure they are 2D tensors
if test_preds_rescaled.dim() == 1:
    test_preds_rescaled = test_preds_rescaled.unsqueeze(1)
    test_targets_rescaled = test_targets_rescaled.unsqueeze(1)

np.save('test_pred', test_preds_rescaled)
np.save('test_target', test_targets_rescaled)

# Compute metrics
test_metrics, test_overall_metrics = compute_metrics(
    test_preds_rescaled, test_targets_rescaled, getattr(test_set, 'mask', None), dataset_name='Test'
)

# Evaluate on val set
val_preds = test_model_jit(
    model=model,
    test_data_loader=val_loader,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True,
    net_params=net_params,
)

# Rescale predictions and targets
val_preds_rescaled = val_preds * std_targets + mean_targets
val_targets_rescaled = torch.cat([data.y for data in val_loader], dim=0) * std_targets + mean_targets

# Ensure they are 2D tensors
if val_preds_rescaled.dim() == 1:
    val_preds_rescaled = val_preds_rescaled.unsqueeze(1)
    val_targets_rescaled = val_targets_rescaled.unsqueeze(1)

np.save('val_pred', val_preds_rescaled)
np.save('val_target', val_targets_rescaled)



# Evaluate on train set
train_preds = test_model_jit(
    model=model,
    test_data_loader=train_loader,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True,
    net_params=net_params,
)


# Rescale predictions and targets
train_preds_rescaled = train_preds * std_targets + mean_targets
train_targets_rescaled = torch.cat([data.y for data in train_loader], dim=0) * std_targets + mean_targets

# Ensure they are 2D tensors
if test_preds_rescaled.dim() == 1:
    test_preds_rescaled = test_preds_rescaled.unsqueeze(1)
    test_targets_rescaled = test_targets_rescaled.unsqueeze(1)

np.save('train_pred', train_preds_rescaled)
np.save('train_target', train_targets_rescaled)



##########################################################################################
#####################    Transfer Learning Functions  ####################################
##########################################################################################

# Function to freeze all layers except the last two
def freeze_model_layers(model):
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last two layers for transfer learning
    for param in model.linear_predict1.parameters():
        param.requires_grad = True
    for param in model.linear_predict2.parameters():
        param.requires_grad = True

##########################################################################################
#####################    Load Transfer Learning Dataset  #################################
##########################################################################################

# Extract configurations for transfer learning dataset
tl_data_config = data_config.get('transfer_learning_dataset', None)

if tl_data_config is not None:
    # Extract parameters
    tl_file_name = tl_data_config.get('file_name')
    tl_file_type = tl_data_config.get('file_type', 'csv')
    tl_smiles_column = tl_data_config.get('smiles_column', 'SMILES')
    tl_target_columns = tl_data_config.get('target_columns', ['Target'])
    tl_global_features_columns = tl_data_config.get('global_features_columns', None)
    tl_split_column = tl_data_config.get('split_column', None)
    tl_sheet_name = tl_data_config.get('sheet_name', 0)
    tl_model_filename = tl_data_config.get('model_save_path', 'gcgat_jitted_GAT_layers_frozen.pth')
    # Load transfer learning dataset
    if tl_file_type == 'csv':
        tl_df = pd.read_csv(tl_file_name)
    elif tl_file_type == 'excel':
        tl_df = pd.read_excel(tl_file_name, sheet_name=tl_sheet_name)
    else:
        raise ValueError(f"Unsupported file type: {tl_file_type}")

    # Read SMILES and targets
    tl_smiles = tl_df[tl_smiles_column].to_numpy()
    tl_targets_df = tl_df[tl_target_columns]
    tl_num_targets = tl_targets_df.shape[1]
    tl_target_names = tl_targets_df.columns.tolist()
    tl_targets = tl_targets_df.to_numpy()  # Shape: (num_samples, num_targets)

    # Read global features if provided
    if tl_global_features_columns is not None:
        tl_global_feats = tl_df[tl_global_features_columns].to_numpy()
    else:
        tl_global_feats = None

    # Handle missing targets
    # TODO: refactor this to a function in utils
    if np.isnan(tl_targets).any():
        # Create mask: True where targets are present
        tl_mask = ~np.isnan(tl_targets)  # Boolean array where True indicates presence
        tl_mask = tl_mask.astype(np.float32)  # Convert to float32 for tensor operations
        # Fill missing targets with zeros (since they'll be masked out during loss computation)
        tl_targets_filled = np.nan_to_num(tl_targets, nan=0.0)
        # Compute mean and std excluding NaNs
        #tl_mean_targets = np.nanmean(tl_targets, axis=0)  # Shape: (num_targets,)
        #tl_std_targets = np.nanstd(tl_targets, axis=0)
        tl_mean_targets = mean_targets
        tl_std_targets = std_targets
        # Standardize targets
        tl_targets_standardized = standardize(tl_targets, tl_mean_targets, tl_std_targets)
        # Replace NaNs with zeros
        tl_targets_standardized_filled = np.nan_to_num(tl_targets_standardized, nan=0.0)
    else:
        tl_mask = None  # No mask needed when all targets are present
        tl_targets_filled = tl_targets
        # Compute mean and std normally
        #tl_mean_targets = np.mean(tl_targets, axis=0)
        #tl_std_targets = np.std(tl_targets, axis=0)
        tl_mean_targets = mean_targets
        tl_std_targets = std_targets
        # Standardize targets
        tl_targets_standardized = standardize(tl_targets, tl_mean_targets, tl_std_targets)
        tl_targets_standardized_filled = tl_targets_standardized

    # Handle custom splits
    if tl_split_column is not None:
        tl_split_values = tl_df[tl_split_column].to_numpy()
        # Map split values to integers
        unique_splits = np.unique(tl_split_values)
        split_mapping = {split: idx for idx, split in enumerate(unique_splits)}
        tl_custom_split = np.array([split_mapping[split] for split in tl_split_values])
    else:
        # If no split column provided, split using ratio
        tl_split_ratios = tl_data_config.get('split_ratios', [0.8, 0.1, 0.1])
        assert sum(tl_split_ratios) == 1.0, "Split ratios must sum to 1.0"
        from sklearn.model_selection import train_test_split

        tl_indices = np.arange(len(tl_smiles))
        tl_train_idx, tl_test_idx = train_test_split(
            tl_indices, test_size=tl_split_ratios[2], random_state=42
        )
        tl_train_idx, tl_val_idx = train_test_split(
            tl_train_idx,
            test_size=tl_split_ratios[1] / (tl_split_ratios[0] + tl_split_ratios[1]),
            random_state=42,
        )
        tl_custom_split = np.zeros(len(tl_smiles), dtype=int)
        tl_custom_split[tl_val_idx] = 1
        tl_custom_split[tl_test_idx] = 2

    # Create Dataset for transfer learning
    tl_data = DataSet(
        smiles=tl_smiles,
        target=tl_targets_standardized_filled,
        global_features=tl_global_feats,
        mask=tl_mask,  # This can be None if no mask is needed
        filter=True,
        fragmentation=fragmentation,  # Use the same fragmentation
        target_columns=tl_target_columns,
        custom_split=tl_custom_split,
    )

    # Split data
    tl_train_set, tl_val_set, tl_test_set = split_data(
        tl_data, split_type='custom', custom_split=tl_data.custom_split
    )

    # Create DataLoaders
    tl_batch_size = data_config.get('batch_size',5000)
    tl_train_loader = DataLoader(tl_train_set, batch_size=tl_batch_size, shuffle=True)
    tl_val_loader = DataLoader(tl_val_set, batch_size=tl_batch_size, shuffle=False)
    tl_test_loader = DataLoader(tl_test_set, batch_size=tl_batch_size, shuffle=False)

##########################################################################################
#####################    Transfer Learning Loop  #########################################
##########################################################################################

    # Function to perform transfer learning on the model
    def transfer_learning(
        base_model,
        train_loader,
        val_loader,
        epochs=100,
        learning_rate=0.001,
        weight_decay=1e-6,
        patience=5,
    ):

        # Clone the base model for transfer learning
        model = copy.deepcopy(base_model)
        model.to(device)

        # Freeze all layers except the last two
        freeze_model_layers(model)

        # Define a new optimizer for the un-frozen parameters
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Set up early stopping
        early_stopper = EarlyStopping(
            patience=patience,
            model_name="transfer_learning_model",
            skip_save=True,
        )

        # Train the model for the specified number of epochs
        train_model_jit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            train_data_loader=train_loader,
            val_data_loader=val_loader,
            epochs=epochs,
            device=device,
            batch_size=tl_batch_size,
            scheduler=scheduler,
            early_stopper=early_stopper,
            model_needs_frag=True,
            net_params=net_params,
        )

        # Save the transfer learned model
        torch.save(model.state_dict(), tl_model_filename)
        print(f"Transfer learned model saved to '{tl_model_filename}'.")

        # Evaluate the model
        val_preds = test_model_jit(
            model=model,
            test_data_loader=val_loader,
            device=device,
            batch_size=tl_batch_size,
            model_needs_frag=True,
            net_params=net_params,
        )
        val_targets_rescaled = (
            torch.cat([data.y for data in val_loader], dim=0) * tl_std_targets + tl_mean_targets
        )
        val_preds_rescaled = val_preds * tl_std_targets + tl_mean_targets

        val_mae = pred_metric(
            prediction=val_preds_rescaled,
            target=val_targets_rescaled,
            metrics='mae',
            print_out=False,
        )['mae']
        print(f"Validation MAE after transfer learning: {val_mae}")

        return model

    # Perform transfer learning
    transfer_learned_model = transfer_learning(
        base_model=model,
        train_loader=tl_train_loader,
        val_loader=tl_val_loader,
        epochs=600,
        learning_rate=0.001,
        weight_decay=1e-6,
        patience=5,
    )

    ##########################################################################################
    #####################    Evaluation on Transfer Learning Test Set ########################
    ##########################################################################################

    # Evaluate on transfer learning test set
    tl_test_preds = test_model_jit(
        model=transfer_learned_model,
        test_data_loader=tl_test_loader,
        device=device,
        batch_size=tl_batch_size,
        model_needs_frag=True,
        net_params=net_params,
    )

    # Rescale predictions and targets
    tl_test_preds_rescaled = tl_test_preds * tl_std_targets + tl_mean_targets
    tl_test_targets_rescaled = (
        torch.cat([data.y for data in tl_test_loader], dim=0) * tl_std_targets + tl_mean_targets
    )

    np.save('tl_test_pred', tl_test_preds_rescaled)
    np.save('tl_test_target', tl_test_targets_rescaled)

    # Evaluate on transfer learning val set
    tl_val_preds = test_model_jit(
        model=transfer_learned_model,
        test_data_loader=tl_val_loader,
        device=device,
        batch_size=tl_batch_size,
        model_needs_frag=True,
        net_params=net_params,
    )

    # Rescale predictions and targets
    tl_val_preds_rescaled = tl_val_preds * tl_std_targets + tl_mean_targets
    tl_val_targets_rescaled = (
            torch.cat([data.y for data in tl_val_loader], dim=0) * tl_std_targets + tl_mean_targets
    )
    np.save('tl_val_pred', tl_val_preds_rescaled)
    np.save('tl_val_target', tl_val_targets_rescaled)

    # Evaluate on transfer learning train set
    tl_train_preds = test_model_jit(
        model=transfer_learned_model,
        test_data_loader=tl_train_loader,
        device=device,
        batch_size=tl_batch_size,
        model_needs_frag=True,
        net_params=net_params,
    )

    # Rescale predictions and targets
    tl_train_preds_rescaled = tl_train_preds * tl_std_targets + tl_mean_targets
    tl_train_targets_rescaled = (
            torch.cat([data.y for data in tl_train_loader], dim=0) * tl_std_targets + tl_mean_targets
    )
    np.save('tl_train_pred', tl_train_preds_rescaled)
    np.save('tl_train_target', tl_train_targets_rescaled)



    # Compute metrics
    tl_test_metrics, tl_test_overall_metrics = compute_metrics(
        tl_test_preds_rescaled,
        tl_test_targets_rescaled,
        getattr(tl_test_set, 'mask', None),
        dataset_name='Transfer Learning Test',
    )
else:
    print("No transfer learning dataset specified in the configuration file.")