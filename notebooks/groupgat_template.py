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

    Parameters:
    - file_name: str, path to the dataset file
    - file_type: 'csv' or 'excel'
    - smiles_column: str, name of the column containing SMILES
    - target_columns: list of str, names of the target columns
    - global_features_columns: list of str or None, names of the global features columns
    - split_column: str or None, name of the column containing split assignments
    - sheet_name: sheet name or index for excel files
    - fragmentation: Fragmentation object or None
    - custom_split_ratios: list of float or None, ratios for train/val/test split if no split column provided
    - mean_targets: numpy array or None, mean of targets for standardization
    - std_targets: numpy array or None, std of targets for standardization
    - default_split: str, default split assignment for molecules not specified in split_column
    - custom_split_mapping: dict or None, mapping of split values to integers (0: train, 1: val, 2: test)
    - use_mean_std_from: DataSet or None, if provided, use mean and std from this DataSet

    Returns:
    - data: DataSet object
    - mean_targets: numpy array, mean of targets
    - std_targets: numpy array, std of targets
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
        # Compute mean and std excluding NaNs if not provided
        if mean_targets is None or std_targets is None:
            mean_targets = np.nanmean(targets, axis=0)  # Shape: (num_targets,)
            std_targets = np.nanstd(targets, axis=0)
        # Standardize targets
        targets_standardized = standardize(targets_filled, mean_targets, std_targets)
    else:
        mask = None  # No mask needed when all targets are present
        targets_filled = targets
        # Compute mean and std normally if not provided
        if mean_targets is None or std_targets is None:
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
        custom_split = np.array([split_mapping.get(split, split_mapping[default_split]) for split in split_values])
    else:
        # If no split column provided, split using ratio
        if custom_split_ratios is not None:
            assert sum(custom_split_ratios) == 1.0, "Split ratios must sum to 1.0"
            from sklearn.model_selection import train_test_split

            indices = np.arange(len(smiles))
            train_idx, test_idx = train_test_split(indices, test_size=custom_split_ratios[2], random_state=42)
            train_idx, val_idx = train_test_split(
                train_idx, test_size=custom_split_ratios[1] / (custom_split_ratios[0] + custom_split_ratios[1]), random_state=42
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
        target=targets_standardized,
        global_features=global_feats,
        mask=mask,  # This can be None if no mask is needed
        filter=True,
        fragmentation=fragmentation,
        target_columns=target_columns,
        custom_split=custom_split,
    )

    return data, mean_targets, std_targets

# Extract configurations for initial dataset
file_name = data_config.get('file_name')
file_type = data_config.get('file_type', 'csv')
smiles_column = data_config.get('smiles_column', 'SMILES')
target_columns = data_config.get('target_columns', ['Target'])
global_features_columns = data_config.get('global_features_columns', None)
split_column = data_config.get('split_column', None)
sheet_name = data_config.get('sheet_name', 0)
fragmentation_settings = data_config.get('fragmentation', None)

# Handle fragmentation
if fragmentation_settings is not None:
    # Get fragmentation settings
    scheme = fragmentation_settings.get('scheme', 'MG_plus_reference')
    save_file_path = fragmentation_settings.get('frag_save_file_path', None)
    tl_save_file_path = fragmentation_settings.get('transfer_learning_frag_save_file_path', None)
    verbose = fragmentation_settings.get('verbose', False)
    print("Initializing fragmentation for main dataset...")
    fragmentation = JT_SubGraph(scheme=scheme, save_file_path=save_file_path, verbose=verbose)
    frag_dim = fragmentation.frag_dim
    print("Done.")

    print("Initializing fragmentation for transfer learning dataset...")
    tl_fragmentation = JT_SubGraph(scheme=scheme, save_file_path=tl_save_file_path, verbose=verbose)
    print("Done.")
else:
    fragmentation = None
    tl_fragmentation = None
    frag_dim = 0

# Load initial dataset using the function
data, mean_targets, std_targets = load_and_preprocess_dataset(
    file_name=file_name,
    file_type=file_type,
    smiles_column=smiles_column,
    target_columns=target_columns,
    global_features_columns=global_features_columns,
    split_column=split_column,
    sheet_name=sheet_name,
    fragmentation=fragmentation,
    custom_split_ratios=data_config.get('split_ratios', [0.8, 0.1, 0.1]),
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
    "num_global_feats": data.global_features.shape[1] if data.global_features is not None else 0,
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

np.save('test_pred', test_preds_rescaled.cpu().numpy())
np.save('test_target', test_targets_rescaled.cpu().numpy())

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

np.save('val_pred', val_preds_rescaled.cpu().numpy())
np.save('val_target', val_targets_rescaled.cpu().numpy())

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
if train_preds_rescaled.dim() == 1:
    train_preds_rescaled = train_preds_rescaled.unsqueeze(1)
    train_targets_rescaled = train_targets_rescaled.unsqueeze(1)

np.save('train_pred', train_preds_rescaled.cpu().numpy())
np.save('train_target', train_targets_rescaled.cpu().numpy())

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
    # Load training data
    tl_training_data_config = tl_data_config.get('training_data')
    tl_train_data, _, _ = load_and_preprocess_dataset(
        file_name=tl_training_data_config.get('file_name'),
        file_type=tl_training_data_config.get('file_type', 'csv'),
        smiles_column=tl_training_data_config.get('smiles_column', 'SMILES'),
        target_columns=tl_training_data_config.get('target_columns', ['Target']),
        global_features_columns=tl_training_data_config.get('global_features_columns', None),
        split_column=tl_training_data_config.get('split_column', None),
        sheet_name=tl_training_data_config.get('sheet_name', 0),
        fragmentation=tl_fragmentation,  # Use tl_fragmentation for training data
        mean_targets=mean_targets,
        std_targets=std_targets,
        default_split='train',
    )

    # Load validation and test data
    tl_val_test_data_config = tl_data_config.get('validation_test_data')
    tl_val_test_data, _, _ = load_and_preprocess_dataset(
        file_name=tl_val_test_data_config.get('file_name'),
        file_type=tl_val_test_data_config.get('file_type', 'csv'),
        smiles_column=tl_val_test_data_config.get('smiles_column', 'SMILES'),
        target_columns=tl_val_test_data_config.get('target_columns', ['Target']),
        global_features_columns=tl_val_test_data_config.get('global_features_columns', None),
        split_column=tl_val_test_data_config.get('split_column', None),
        sheet_name=tl_val_test_data_config.get('sheet_name', 0),
        fragmentation=fragmentation,  # Use original fragmentation for val/test data
        mean_targets=mean_targets,
        std_targets=std_targets,
        default_split='train',
    )

    # Combine datasets properly
    # Concatenate the data lists and other attributes
    combined_smiles = np.concatenate([tl_train_data.smiles, tl_val_test_data.smiles])
    combined_targets = np.concatenate([tl_train_data.target, tl_val_test_data.target])
    combined_global_feats = None
    if tl_train_data.global_features is not None and tl_val_test_data.global_features is not None:
        combined_global_feats = np.concatenate([tl_train_data.global_features, tl_val_test_data.global_features])
    combined_masks = None
    if tl_train_data.mask is not None and tl_val_test_data.mask is not None:
        combined_masks = np.concatenate([tl_train_data.mask, tl_val_test_data.mask])
    combined_custom_split = np.concatenate([tl_train_data.custom_split, tl_val_test_data.custom_split])

    # Create a new DataSet with the combined data
    breakpoint()
    tl_data = DataSet(
        smiles=combined_smiles,
        target=combined_targets.squeeze(),
        global_features=combined_global_feats,
        mask=combined_masks,
        filter=False,
        fragmentation=None,  # No need to fragment again
        custom_split=combined_custom_split,
    )
    combined_graphs = tl_train_data.graphs + tl_val_test_data.graphs
    tl_data.graphs = combined_graphs

    # Manually set the fragmentation data for each data point
    # Since we used different fragmentation objects, we need to ensure the fragmentation data is correctly set
    #tl_data.fragmentation_data = tl_train_data.fragmentation_data + tl_val_test_data.fragmentation_data

    # Split data
    tl_train_set, tl_val_set, tl_test_set = split_data(
        tl_data, split_type='custom', custom_split=tl_data.custom_split
    )

    # Create DataLoaders
    tl_batch_size = data_config.get('batch_size', 5000)
    tl_train_loader = DataLoader(tl_train_set, batch_size=tl_batch_size, shuffle=True)
    tl_val_loader = DataLoader(tl_val_set, batch_size=tl_batch_size, shuffle=False)
    tl_test_loader = DataLoader(tl_test_set, batch_size=tl_batch_size, shuffle=False)

    tl_model_filename = tl_data_config.get('model_save_path', 'gcgat_jitted_GAT_layers_frozen.pth')

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
            torch.cat([data.y for data in val_loader], dim=0) * std_targets + mean_targets
        )
        val_preds_rescaled = val_preds * std_targets + mean_targets

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
    tl_test_preds_rescaled = tl_test_preds * std_targets + mean_targets
    tl_test_targets_rescaled = (
        torch.cat([data.y for data in tl_test_loader], dim=0) * std_targets + mean_targets
    )

    np.save('tl_test_pred', tl_test_preds_rescaled.cpu().numpy())
    np.save('tl_test_target', tl_test_targets_rescaled.cpu().numpy())

    # Compute metrics
    tl_test_metrics, tl_test_overall_metrics = compute_metrics(
        tl_test_preds_rescaled,
        tl_test_targets_rescaled,
        getattr(tl_test_set, 'mask', None),
        dataset_name='Transfer Learning Test',
    )

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
    tl_val_preds_rescaled = tl_val_preds * std_targets + mean_targets
    tl_val_targets_rescaled = (
            torch.cat([data.y for data in tl_val_loader], dim=0) * std_targets + mean_targets
    )
    np.save('tl_val_pred', tl_val_preds_rescaled.cpu().numpy())
    np.save('tl_val_target', tl_val_targets_rescaled.cpu().numpy())

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
    tl_train_preds_rescaled = tl_train_preds * std_targets + mean_targets
    tl_train_targets_rescaled = (
            torch.cat([data.y for data in tl_train_loader], dim=0) * std_targets + mean_targets
    )
    np.save('tl_train_pred', tl_train_preds_rescaled.cpu().numpy())
    np.save('tl_train_target', tl_train_targets_rescaled.cpu().numpy())

    # Compute metrics for transfer learning train set
    tl_train_metrics, tl_train_overall_metrics = compute_metrics(
        tl_train_preds_rescaled,
        tl_train_targets_rescaled,
        getattr(tl_train_set, 'mask', None),
        dataset_name='Transfer Learning Train',
    )

    # Compute metrics for transfer learning validation set
    tl_val_metrics, tl_val_overall_metrics = compute_metrics(
        tl_val_preds_rescaled,
        tl_val_targets_rescaled,
        getattr(tl_val_set, 'mask', None),
        dataset_name='Transfer Learning Validation',
    )

else:
    print("No transfer learning dataset specified in the configuration file.")
