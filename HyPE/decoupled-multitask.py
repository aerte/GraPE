# Import necessary libraries
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
root = './env/params_prediction.xlsx'
sheet_name = ''

df = pd.read_excel(root)
# Read SMILES and target properties
smiles = df['SMILES'].to_numpy()
target_columns = ['A0', 'B0', 'C0', 'D0', 'E0']
targets_df = df[target_columns]  # You can have any number of targets here
num_targets = targets_df.shape[1]
target_names = targets_df.columns.tolist()
targets = targets_df.to_numpy()  # Shape: (num_samples, num_targets)

# Read tags for custom splits
tags = df['bin'].to_numpy()
unique_tags = np.unique(tags)
tag_to_int = {'train': 0, 'val': 1, 'test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

# Global features
#global_feats = df['T'].to_numpy()

# Standardize targets
mean_targets = np.mean(targets, axis=0)  # Shape: (num_targets,)
std_targets = np.std(targets, axis=0)
targets_standardized = standardize(targets, mean_targets, std_targets)

# Standardize global features
# mean_global_feats = np.mean(global_feats)
# std_global_feats = np.std(global_feats)
# global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

########################## Fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("Initializing fragmentation...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme)
frag_dim = fragmentation.frag_dim
print("Done.")

########################### DataSet Creation ######################################

data = DataSet(
    smiles=smiles,
    target=targets_standardized,
    filter=True,
    fragmentation=fragmentation,
    target_columns=target_columns,
)

# Split data using custom splits
train, val, test = split_data(data, split_type='custom', custom_split=custom_split)

############################################################################################

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
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
model_filename = 'gcgat_jitted_latest_coupled_multitask.pth'

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
def calculate_mae(preds_rescaled, targets_rescaled, dataset_name='Test'):
    maes = []
    for i, prop in enumerate(target_columns):
        pred_prop = preds_rescaled[:, i]
        target_prop = targets_rescaled[:, i]
        mae = torch.mean(torch.abs(pred_prop - target_prop))
        print(f'MAE for property {prop} on {dataset_name} set: {mae.item():.4f}')
        maes.append(mae.item())
    overall_mae = torch.mean(torch.abs(preds_rescaled - targets_rescaled))
    print(f'Overall MAE across all properties on {dataset_name} set: {overall_mae.item():.4f}')
    return maes, overall_mae.item()

def compute_metrics(preds_rescaled, targets_rescaled, dataset_name='Test'):
    metrics_per_property = []
    for i, prop in enumerate(target_columns):
        pred_prop = preds_rescaled[:, i]
        target_prop = targets_rescaled[:, i]
        # Call pred_metric to compute desired metrics
        results = pred_metric(pred_prop, target_prop, metrics='all', print_out=False)
        print(f"Metrics for property {prop} on {dataset_name} set:")
        for metric_name, value in results.items():
            print(f"{metric_name.upper()}: {value:.4f}")
        metrics_per_property.append(results)
    # Compute overall metrics
    overall_results = pred_metric(preds_rescaled, targets_rescaled, metrics='all', print_out=False)
    print(f"Overall metrics across all properties on {dataset_name} set:")
    for metric_name, value in overall_results.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    return metrics_per_property, overall_results

# Calculate metrics for the test set
test_maes, test_overall_mae = calculate_mae(preds_rescaled, targets_rescaled, dataset_name='Test')

####### Example for overall evaluation of the MAE #########
train_preds = test_model_jit(model=model, test_data_loader=train, device=device, batch_size=batch_size)
val_preds = test_model_jit(model=model, test_data_loader=val, device=device, batch_size=batch_size)

# Rescale predictions and targets
train_preds_rescaled = train_preds * std_targets + mean_targets
train_targets_rescaled = train.y * std_targets + mean_targets

val_preds_rescaled = val_preds * std_targets + mean_targets
val_targets_rescaled = val.y * std_targets + mean_targets

# Calculate metrics for train and validation sets
train_maes, train_overall_mae = calculate_mae(train_preds_rescaled, train_targets_rescaled, dataset_name='Train')
val_maes, val_overall_mae = calculate_mae(val_preds_rescaled, val_targets_rescaled, dataset_name='Validation')

# Calculate overall MAE across all datasets
overall_mae_per_property = []
for i in range(num_targets):
    overall_mae = (train_maes[i] + val_maes[i] + test_maes[i]) / 3
    overall_prop = target_columns[i]
    print(f'Overall MAE for property {overall_prop}: {overall_mae:.4f}')
    overall_mae_per_property.append(overall_mae)

# Overall MAE across all properties and datasets
overall_mae_all = (train_overall_mae + val_overall_mae + test_overall_mae) / 3
print(f'Overall MAE across all properties and datasets: {overall_mae_all:.4f}')

# Modify the test_model function to return both predictions and targets
def test_model_jit_with_parity(
    model: torch.nn.Module,
    test_data_loader: Union[List, DataLoader],
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
                            motif_nodes,
                            global_feats,
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
                            motif_nodes,
                            global_feats,
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

train_targets = train_targets.view(-1, num_targets)
val_targets = val_targets.view(-1, num_targets)
test_targets = test_targets.view(-1, num_targets)

train_preds = train_preds.view(-1, num_targets)
val_preds = val_preds.view(-1, num_targets)
test_preds = test_preds.view(-1, num_targets)

####### Rescale predictions and targets #########
train_preds_rescaled = train_preds * std_targets + mean_targets
train_targets_rescaled = train_targets * std_targets + mean_targets

val_preds_rescaled = val_preds * std_targets + mean_targets
val_targets_rescaled = val_targets * std_targets + mean_targets

test_preds_rescaled = test_preds * std_targets + mean_targets
test_targets_rescaled = test_targets * std_targets + mean_targets

test_metrics_per_property, test_overall_metrics = compute_metrics(preds_rescaled, targets_rescaled, dataset_name='Test')

####### Example for overall evaluation of the metrics #########
train_preds = test_model_jit(model=model, test_data_loader=train, device=device, batch_size=batch_size)
val_preds = test_model_jit(model=model, test_data_loader=val, device=device, batch_size=batch_size)

# Reshape predictions and targets
train_preds = train_preds.view(-1, num_targets)

ty = torch.from_numpy(train.y) #messy, quick hack
train_targets_rescaled = ty.view(-1, num_targets) * std_targets + mean_targets
train_preds_rescaled = train_preds * std_targets + mean_targets

vy = torch.from_numpy(val.y)
val_targets_rescaled = vy.view(-1, num_targets) * std_targets + mean_targets
val_preds_rescaled = val_preds * std_targets + mean_targets

# Calculate metrics for train and validation sets
train_metrics_per_property, train_overall_metrics = compute_metrics(train_preds_rescaled, train_targets_rescaled, dataset_name='Train')
val_metrics_per_property, val_overall_metrics = compute_metrics(val_preds_rescaled, val_targets_rescaled, dataset_name='Validation')

# Calculate overall metrics across all datasets
overall_metrics_per_property = []
for i in range(num_targets):
    prop = target_columns[i]
    # Extract metrics for this property from each dataset
    train_metrics = train_metrics_per_property[i]
    val_metrics = val_metrics_per_property[i]
    test_metrics = test_metrics_per_property[i]
    # Compute average of each metric
    avg_metrics = {}
    for metric_name in train_metrics.keys():
        avg_value = (train_metrics[metric_name] + val_metrics[metric_name] + test_metrics[metric_name]) / 3
        avg_metrics[metric_name] = avg_value
    overall_metrics_per_property.append(avg_metrics)
    # Print the overall metrics for this property
    print(f'Overall metrics for property {prop}:')
    for metric_name, value in avg_metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")

# Overall metrics across all properties and datasets
overall_metrics_across_all = {}
for metric_name in train_overall_metrics.keys():
    overall_value = (train_overall_metrics[metric_name] + val_overall_metrics[metric_name] + test_overall_metrics[metric_name]) / 3
    overall_metrics_across_all[metric_name] = overall_value
print('Overall metrics across all properties and datasets:')
for metric_name, value in overall_metrics_across_all.items():
    print(f"{metric_name.upper()}: {value:.4f}")

####### Creating Parity Plot #########
def create_parity_plot(
    train_preds_rescaled, train_targets_rescaled,
    val_preds_rescaled, val_targets_rescaled,
    test_preds_rescaled, test_targets_rescaled,
    target_columns
):
    import matplotlib.pyplot as plt

    for i, prop in enumerate(target_columns):
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
        num_train = len(train_preds_rescaled)
        num_val = len(val_preds_rescaled)
        num_test = len(test_preds_rescaled)
        train_labels = np.array(['Train'] * num_train)
        val_labels = np.array(['Validation'] * num_val)
        test_labels = np.array(['Test'] * num_test)
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

# Generate parity plots
create_parity_plot(
    train_preds_rescaled, train_targets_rescaled,
    val_preds_rescaled, val_targets_rescaled,
    test_preds_rescaled, test_targets_rescaled,
    target_columns
)


##############################################################################################
################                         Grokking                  ###########################
##############################################################################################
if False:
    from typing import Callable, Union, List, Tuple, Optional
    from torch.optim import Optimizer
    import torch.nn.functional as F
    from torch_geometric.data import Data

    def train_model_jit_grokk(
        model: torch.nn.Module,
        loss_func: Union[Callable, str],
        optimizer: Optimizer,
        train_data_loader: Union[List[Data], DataLoader],
        val_data_loader: Union[List[Data], DataLoader],
        device: str = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopper=None,
        scheduler: lr_scheduler._LRScheduler = None,
        tuning: bool = False,
        model_name: str = None,
        model_needs_frag: bool = False,
        net_params: dict = None,
        alpha: float = 0.98,
        lamb: float = 2.0,
    ) -> tuple[List[float], List[float]]:
        """
        Training function adapted for the JIT-compiled model, which requires individual tensors as input.
        This version incorporates the Grokfast method (gradfilter_ema).
        """
        loss_functions = {
            'mse': F.mse_loss,
            'mae': F.l1_loss
        }

        if isinstance(loss_func, str):
            loss_func = loss_functions[loss_func]

        device = torch.device('cpu') if device is None else torch.device(device)

        exclude_keys = None
        # Exclude fragmentation keys if the model doesn't need them
        if not model_needs_frag:
            if hasattr(train_data_loader, "fragmentation"):
                if train_data_loader.fragmentation is not None:
                    exclude_keys = ["frag_graphs", "motif_graphs"]

        if not isinstance(train_data_loader, DataLoader):
            train_data_loader = DataLoader(
                train_data_loader, batch_size=batch_size, exclude_keys=exclude_keys
            )

        if not isinstance(val_data_loader, DataLoader):
            val_data_loader = DataLoader(
                val_data_loader, batch_size=batch_size, exclude_keys=exclude_keys
            )

        model.train()
        train_loss = []
        val_loss = []

        # Initialize grads to None before the training loop
        grads = None

        def handle_heterogenous_sizes(y, out):
            if not isinstance(out, torch.Tensor):
                return out
            if y.dim() == out.dim():
                return out
            return out.squeeze()  # Needed for some models

        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                temp = np.zeros(len(train_data_loader))
                for idx, batch in enumerate(train_data_loader):
                    optimizer.zero_grad()
                    # Extract tensors from batch
                    data_x = batch.x.to(device)
                    data_edge_index = batch.edge_index.to(device)
                    data_edge_attr = batch.edge_attr.to(device)
                    data_batch = batch.batch.to(device)

                    # Fragment graphs
                    frag_graphs = batch.frag_graphs  # List[Data]
                    frag_batch_list = []
                    frag_x_list = []
                    frag_edge_index_list = []
                    frag_edge_attr_list = []
                    node_offset = 0
                    for i, frag in enumerate(frag_graphs):
                        num_nodes = frag.num_nodes
                        frag_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                        frag_x_list.append(frag.x.to(device))
                        adjusted_edge_index = frag.edge_index + node_offset
                        frag_edge_index_list.append(adjusted_edge_index.to(device))
                        frag_edge_attr_list.append(frag.edge_attr.to(device))
                        node_offset += num_nodes

                    frag_x = torch.cat(frag_x_list, dim=0)
                    frag_edge_index = torch.cat(frag_edge_index_list, dim=1)
                    frag_edge_attr = torch.cat(frag_edge_attr_list, dim=0)
                    frag_batch = Batch.from_data_list(batch.frag_graphs).to(device).batch
                    motif_nodes = Batch.from_data_list(batch.frag_graphs).x.to(device)

                    # Junction graphs (motif graphs)
                    junction_graphs = batch.motif_graphs  # List[Data]
                    junction_batch_list = []
                    junction_x_list = []
                    junction_edge_index_list = []
                    junction_edge_attr_list = []
                    node_offset = 0
                    for i, motif in enumerate(junction_graphs):
                        num_nodes = motif.num_nodes
                        junction_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                        junction_x_list.append(motif.x.to(device))
                        adjusted_edge_index = motif.edge_index + node_offset
                        junction_edge_index_list.append(adjusted_edge_index.to(device))
                        junction_edge_attr_list.append(motif.edge_attr.to(device))
                        node_offset += num_nodes

                    junction_x = torch.cat(junction_x_list, dim=0)
                    junction_edge_index = torch.cat(junction_edge_index_list, dim=1)
                    junction_edge_attr = torch.cat(junction_edge_attr_list, dim=0)
                    junction_batch = torch.cat(junction_batch_list, dim=0)

                    if hasattr(batch, 'global_feats') and batch.global_feats is not None:
                        global_feats = batch.global_feats.to(device)
                    else:
                        num_mols = data_batch.max().item() + 1  # Number of molecules in the batch
                        global_feats = torch.zeros((num_mols, 1), device=device)  # Singleton per molecule

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
                        motif_nodes,
                        global_feats,
                    )

                    out = handle_heterogenous_sizes(batch.y.to(device), out)
                    by = batch.y.view(out.shape[0], out.shape[1]).to(device)
                    loss_train = loss_func(by, out)

                    temp[idx] = loss_train.detach().cpu().numpy()

                    loss_train.backward()

                    optimizer.step()

                loss_train = np.mean(temp)
                train_loss.append(loss_train)

                # Validation loop
                model.eval()
                temp = np.zeros(len(val_data_loader))
                with torch.no_grad():
                    for idx, batch in enumerate(val_data_loader):
                        data_x = batch.x.to(device)
                        data_edge_index = batch.edge_index.to(device)
                        data_edge_attr = batch.edge_attr.to(device)
                        data_batch = batch.batch.to(device)

                        # Fragment graphs
                        frag_graphs = batch.frag_graphs  # List[Data]
                        frag_batch_list = []
                        frag_x_list = []
                        frag_edge_index_list = []
                        frag_edge_attr_list = []
                        node_offset = 0
                        for i, frag in enumerate(frag_graphs):
                            num_nodes = frag.num_nodes
                            frag_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                            frag_x_list.append(frag.x.to(device))
                            adjusted_edge_index = frag.edge_index + node_offset
                            frag_edge_index_list.append(adjusted_edge_index.to(device))
                            frag_edge_attr_list.append(frag.edge_attr.to(device))
                            node_offset += num_nodes

                        frag_x = torch.cat(frag_x_list, dim=0)
                        frag_edge_index = torch.cat(frag_edge_index_list, dim=1)
                        frag_edge_attr = torch.cat(frag_edge_attr_list, dim=0)
                        frag_batch = Batch.from_data_list(batch.frag_graphs).to(device).batch
                        motif_nodes = Batch.from_data_list(batch.frag_graphs).x.to(device)

                        # Junction graphs (motif graphs)
                        junction_graphs = batch.motif_graphs  # List[Data]
                        junction_batch_list = []
                        junction_x_list = []
                        junction_edge_index_list = []
                        junction_edge_attr_list = []
                        node_offset = 0
                        for i, motif in enumerate(junction_graphs):
                            num_nodes = motif.num_nodes
                            junction_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
                            junction_x_list.append(motif.x.to(device))
                            adjusted_edge_index = motif.edge_index + node_offset
                            junction_edge_index_list.append(adjusted_edge_index.to(device))
                            junction_edge_attr_list.append(motif.edge_attr.to(device))
                            node_offset += num_nodes

                        junction_x = torch.cat(junction_x_list, dim=0)
                        junction_edge_index = torch.cat(junction_edge_index_list, dim=1)
                        junction_edge_attr = torch.cat(junction_edge_attr_list, dim=0)
                        junction_batch = torch.cat(junction_batch_list, dim=0)

                        if hasattr(batch, 'global_feats') and batch.global_feats is not None:
                            global_feats = batch.global_feats.to(device)
                        else:
                            global_feats = torch.empty(0).to(device)  # Can't have optional args in JIT

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
                            motif_nodes,
                            global_feats,
                        )

                        out = handle_heterogenous_sizes(batch.y.to(device), out)
                        temp[idx] = loss_func(batch.y.view(out.shape[0], out.shape[1]).to(device), out).detach().cpu().numpy()

                loss_val = np.mean(temp)
                val_loss.append(loss_val)
                model.train()  # Switch back to training mode

                if epoch % 2 == 0:
                    pbar.set_description(f"Epoch {epoch}, Training Loss: {loss_train:.3f}, Validation Loss: {loss_val:.3f}")

                if scheduler is not None:
                    scheduler.step(loss_val)

                if early_stopper is not None:
                    early_stopper(val_loss=loss_val, model=model)
                    if early_stopper.stop:
                        print("Early stopping reached with best validation loss: {:.4f}".format(early_stopper.best_score))
                        early_stopper.stop_epoch = epoch - early_stopper.patience
                        if tuning:
                            pass
                        else:
                            break

                pbar.update(1)
            if early_stopper and not early_stopper.stop and model_name:
                torch.save(model.state_dict(), model_name)
                print(f'Model saved at: {model_name}')

        return train_loss, val_loss