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
from jinja2 import Template

# Function to standardize data
def standardize(x, mean, std):
    return (x - mean) / std

# Set seed for reproducibility
set_seed(42)

##########################################################################################
#####################    Data Input and Preprocessing  ###################################
##########################################################################################

# Define the variables to be used in the template
with open('config.yaml', 'r') as f:
    f_config_unparsed = yaml.safe_load(f)
    template_vars = {
        'root_path': f_config_unparsed.get('root_path')
    }

# Load and render the YAML template
with open('config.yaml', 'r') as f:
    template = Template(f.read())
    rendered_yaml = template.render(template_vars)

# Now parse the rendered YAML
data_config = yaml.safe_load(rendered_yaml)

# Function to load and preprocess dataset
def load_and_preprocess_dataset(
    file_name,
    file_type='csv',
    smiles_column='SMILES',
    target_columns=['Target'],
    global_features_columns=None,
    split_column=None,
    sheet_name=0,
    fragmentation=None,
    custom_split_ratios=None,
    mean_targets=None,
    std_targets=None,
    default_split='train',
    custom_split_mapping=None,
    use_mean_std_from=None,
):
    """
    Loads and preprocesses a dataset.

    Returns:
    - data: DataSet object
    - mean_targets: numpy array, mean of targets
    - std_targets: numpy array, std of targets
    - target_columns: list of str, names of the target columns
    """
    # Load the dataset
    if file_type == 'csv':
        df = pd.read_csv(file_name)
    elif file_type == 'excel':
        df = pd.read_excel(file_name,)
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
    if np.isnan(targets).any():
        # Create mask: True where targets are present
        mask = ~np.isnan(targets)  # Boolean array where True indicates presence
        mask = mask.astype(np.float32)  # Convert to float32 for tensor operations
        # Fill missing targets with zeros (since they'll be masked out during loss computation)
        targets_filled = np.nan_to_num(targets, nan=0.0)
        # Use provided mean and std if available
        if mean_targets is not None and std_targets is not None:
            # Standardize targets using provided mean and std
            targets_standardized = standardize(targets_filled, mean_targets, std_targets)
        else:
            # Compute mean and std excluding NaNs
            mean_targets = np.nanmean(targets, axis=0)  # Shape: (num_targets,)
            std_targets = np.nanstd(targets, axis=0)
            # Standardize targets
            targets_standardized = standardize(targets_filled, mean_targets, std_targets)
    else:
        mask = None  # No mask needed when all targets are present
        targets_filled = targets
        if mean_targets is not None and std_targets is not None:
            # Standardize targets using provided mean and std
            targets_standardized = standardize(targets_filled, mean_targets, std_targets)
        else:
            # Compute mean and std normally
            mean_targets = np.mean(targets, axis=0)
            std_targets = np.std(targets, axis=0)
            # Standardize targets
            targets_standardized = standardize(targets, mean_targets, std_targets)

    # Handle fragmentation
    # Note: fragmentation should be provided
    if fragmentation is not None:
        frag_dim = fragmentation.frag_dim
    else:
        frag_dim = 0

    # Handle custom splits
    if split_column is not None and split_column in df.columns:
        split_values = df[split_column].fillna(default_split).to_numpy()
        # Map split values to integers
        if custom_split_mapping is not None:
            split_mapping = custom_split_mapping
        else:
            split_mapping = {'train': 0, 'val': 1, 'test': 2}
        # Convert split values to lower case for matching
        custom_split = np.array([split_mapping.get(str(split).lower(), split_mapping[default_split]) for split in split_values])
    else:
        # If no split column provided, split using ratio
        if custom_split_ratios is not None:
            assert sum(custom_split_ratios) == 1.0, "Split ratios must sum to 1.0"
            from sklearn.model_selection import train_test_split

            indices = np.arange(len(smiles))
            train_size = custom_split_ratios[0]
            val_size = custom_split_ratios[1]
            test_size = custom_split_ratios[2]

            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=42
            )
            train_idx, val_idx = train_test_split(
                train_idx, test_size=val_size / (train_size + val_size), random_state=42
            )

            custom_split = np.full(len(smiles), fill_value=0, dtype=int)
            custom_split[val_idx] = 1
            custom_split[test_idx] = 2
        else:
            # Assign all to default_split
            custom_split = np.full(len(smiles), fill_value=0, dtype=int)  # Default to 'train'

    # Create Dataset
    data = DataSet(
        smiles=smiles,
        target=targets_standardized.squeeze(),
        global_features=None,
        mask=mask,  # This can be None if no mask is needed
        filter=True,
        fragmentation=fragmentation,
        custom_split=custom_split,
    )

    return data, mean_targets, std_targets, target_columns

# Extract fragmentation settings
fragmentation_settings = data_config.get('fragmentation', None)
# Handle fragmentation
if fragmentation_settings is not None:
    # Get fragmentation settings
    scheme = fragmentation_settings.get('scheme', 'MG_plus_reference')
    save_file_path = fragmentation_settings.get('frag_save_file_path', None)
    tl_save_file_path = fragmentation_settings.get('transfer_learning_frag_save_file_path', None)
    verbose = fragmentation_settings.get('verbose', False)

    # Initialize fragmentation for training data
    fragmentation = JT_SubGraph(scheme=scheme, save_file_path=save_file_path, verbose=verbose)
    frag_dim = fragmentation.frag_dim

    # Initialize fragmentation for validation/test data
    tl_fragmentation = JT_SubGraph(scheme=scheme, save_file_path=tl_save_file_path, verbose=verbose)

else:
    fragmentation = None
    tl_fragmentation = None
    frag_dim = 0

# Load initial training data
initial_training_data_config = data_config.get('transfer_learning_dataset', {}).get('training_data', {})
initial_validation_data_config = data_config.get('transfer_learning_dataset', {}).get('validation_test_data', {})

# Load training data (icas_omega.xlsx) - using fragmentation
train_data, mean_targets, std_targets, target_columns = load_and_preprocess_dataset(
    file_name=initial_training_data_config.get('file_name'),
    file_type=initial_training_data_config.get('file_type', 'excel'),
    smiles_column=initial_training_data_config.get('smiles_column', 'SMILES'),
    target_columns=initial_training_data_config.get('target_columns', ['Target']),
    global_features_columns=initial_training_data_config.get('global_features_columns', None),
    split_column=initial_training_data_config.get('split_column', None),
    sheet_name=initial_training_data_config.get('sheet_name', 0),
    fragmentation=fragmentation,  # Use fragmentation with save_file_path for training data
    custom_split_ratios=initial_training_data_config.get('split_ratios', [1.0, 0.0, 0.0]),
    default_split='train',  # All data as training data
)

# Now, when loading validation/test data, we use mean_targets and std_targets from train_data
# Load validation data (omega_splits.xlsx) - using tl_fragmentation
val_data, _, _, _ = load_and_preprocess_dataset(
    file_name=initial_validation_data_config.get('file_name'),
    file_type=initial_validation_data_config.get('file_type', 'excel'),
    smiles_column=initial_validation_data_config.get('smiles_column', 'Smiles'),
    target_columns=initial_validation_data_config.get('target_columns', ['Target']),
    global_features_columns=initial_validation_data_config.get('global_features_columns', None),
    split_column=initial_validation_data_config.get('split_column', 'Split'),
    sheet_name=initial_validation_data_config.get('sheet_name', 0),
    fragmentation=tl_fragmentation,  # Use tl_fragmentation with its own save_file_path
    custom_split_ratios=initial_validation_data_config.get('split_ratios', [0.8, 0.1, 0.1]),
    default_split='train',
    mean_targets=mean_targets,  # Use mean and std from synthetic data
    std_targets=std_targets,
)

# Split val_data into train, val, test sets
omega_train_set, omega_val_set, omega_test_set = split_data(val_data, split_type='custom', custom_split=val_data.custom_split)

# For initial training, use train_data as train_set, and omega_val_set as val_set
train_set = train_data
val_set = omega_val_set  # Use omega validation set for initial validation
test_set = None  # No test set during initial training

# Create DataLoaders
batch_size = data_config.get('batch_size', 5000)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

##########################################################################################
#####################    Model Setup and Training  #######################################
##########################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define network parameters
# Extract model parameters from the config file
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
    "num_global_feats": train_data.global_features.shape[1] if train_data.global_features is not None else 0,
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
    "output_dim": len(target_columns),  # Number of target properties
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
tl_model_filename = data_config.get('transfer_learning_dataset', {}).get('model_save_path', 'transfer_learning_omega_model.pth')

# Load model if saved
if os.path.exists(model_filename):
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
else:
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

##########################################################################################
#####################    Evaluation and Metrics After Initial Training ###################
##########################################################################################

# Evaluate the model on the validation set
val_preds = test_model_jit(
    model=model,
    test_data_loader=val_loader,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True,
    net_params=net_params,
)

# Rescale predictions and targets using mean and std from synthetic data
val_preds_rescaled = val_preds * std_targets + mean_targets
val_targets_rescaled = (
    torch.cat([data.y for data in val_loader], dim=0) * std_targets + mean_targets
)

# Compute metrics
def compute_metrics(preds_rescaled, targets_rescaled, mask=None, dataset_name='Test', target_columns=None):
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
        target_prop = targets_rescaled
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

# Compute and print metrics after initial training
print("Metrics after training full model on synthetic data and validating on real data:")
initial_val_metrics, initial_val_overall_metrics = compute_metrics(
    val_preds_rescaled,
    val_targets_rescaled,
    getattr(val_set, 'mask', None),
    dataset_name='Validation',
    target_columns=target_columns,
)

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
#####################    Transfer Learning  ##############################################
##########################################################################################

# Create DataLoaders for transfer learning
tl_batch_size = data_config.get('batch_size', 5000)
tl_train_loader = DataLoader(omega_train_set, batch_size=tl_batch_size, shuffle=True)
tl_val_loader = DataLoader(omega_val_set, batch_size=tl_batch_size, shuffle=False)
tl_test_loader = DataLoader(omega_test_set, batch_size=tl_batch_size, shuffle=False)

# Function to perform transfer learning on the model
def transfer_learning(
    base_model,
    train_loader,
    val_loader,
    epochs=100,
    learning_rate=0.001,
    weight_decay=1e-6,
    patience=60,
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

    # Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, min_lr=1.00E-09, patience=15
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

    return model

# Perform transfer learning using omega_train_set and omega_val_set
transfer_learned_model = transfer_learning(
    base_model=model,
    train_loader=tl_train_loader,
    val_loader=tl_val_loader,
    epochs=600,
    learning_rate=0.001,
    weight_decay=1e-6,
    patience=40,
)

##########################################################################################
#####################    Evaluation After Transfer Learning ##############################
##########################################################################################

# Evaluate on transfer learning validation set
tl_val_preds = test_model_jit(
    model=transfer_learned_model,
    test_data_loader=tl_val_loader,
    device=device,
    batch_size=tl_batch_size,
    model_needs_frag=True,
    net_params=net_params,
)

tl_val_preds_rescaled = tl_val_preds * std_targets + mean_targets  # Use mean and std from synthetic data
tl_val_targets_rescaled = (
    torch.cat([data.y for data in tl_val_loader], dim=0) * std_targets + mean_targets
)

# Compute and print metrics after transfer learning on validation set
print("\nMetrics after transfer learning of last layers on real dataset (Validation Set):")
tl_val_metrics, tl_val_overall_metrics = compute_metrics(
    tl_val_preds_rescaled,
    tl_val_targets_rescaled,
    getattr(omega_val_set, 'mask', None),
    dataset_name='Validation',
    target_columns=target_columns,
)

# Evaluate on transfer learning test set
tl_test_preds = test_model_jit(
    model=transfer_learned_model,
    test_data_loader=tl_test_loader,
    device=device,
    batch_size=tl_batch_size,
    model_needs_frag=True,
    net_params=net_params,
)

tl_test_preds_rescaled = tl_test_preds * std_targets + mean_targets  # Use mean and std from synthetic data
tl_test_targets_rescaled = (
    torch.cat([data.y for data in tl_test_loader], dim=0) * std_targets + mean_targets
)

# Compute and print metrics after transfer learning on test set
print("\nMetrics after transfer learning of last layers on real dataset (Test Set):")
tl_test_metrics, tl_test_overall_metrics = compute_metrics(
    tl_test_preds_rescaled,
    tl_test_targets_rescaled,
    getattr(omega_test_set, 'mask', None),
    dataset_name='Test',
    target_columns=target_columns,
)