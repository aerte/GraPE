from grape_chem.models import GroupGAT_jittable
from grape_chem.utils import (
    DataSet, train_model_jit, EarlyStopping, split_data, 
    test_model_jit, pred_metric, return_hidden_layers, set_seed, JT_SubGraph
)
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import DataLoader, Data, Batch
from tqdm import tqdm
from typing import Union, List, Tuple
from torch import Tensor
import os

# Ensure GraPE is installed
# pip install "git+https://github.com/aerte/GraPE.git#subdirectory=python"

def standardize(x, mean, std):
    return (x - mean) / std

# Set random seed for reproducibility
set_seed(42)

# Hyperparameters
epochs = 1000
batch_size = 1500
patience = 40
hidden_dim = 47
learning_rate = 0.001
weight_decay = 1e-4
mlp_layers = 2
atom_layers = 3
mol_layers = 3

# Data Loading
root = './env/ICP.xlsx'
df = pd.read_excel(root)
smiles = df['SMILES'].to_numpy()
target = df['Value'].to_numpy()
tags = df['Subset'].to_numpy()

# Custom Split
tag_to_int = {'Training': 0, 'Validation': 1, 'Test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

# Global Features
global_feats = df['T'].to_numpy()

# Standardize Target and Global Features
mean_target, std_target = np.mean(target), np.std(target)
target = standardize(target, mean_target, std_target)
mean_global_feats, std_global_feats = np.mean(global_feats), np.std(global_feats)
global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

# Fragmentation
fragmentation_scheme = "MG_plus_reference"
fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path="env/default_ICP_frag_pdd")
frag_dim = fragmentation.frag_dim

# Create Dataset
data = DataSet(
    smiles=smiles, 
    target=target, 
    global_features=global_feats, 
    filter=True, 
    fragmentation=fragmentation
)

# Split Data
train, val, test = split_data(data, split_type='custom', custom_split=custom_split)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Parameters
mlp = return_hidden_layers(mlp_layers)
net_params = {
    "device": device,
    "num_atom_type": 44,
    "num_bond_type": 12,
    "dropout": 0.0,
    "MLP_layers": mlp_layers,
    "frag_dim": frag_dim,
    "final_dropout": 0.119,
    "use_global_features": True,
    "num_heads": 1,
    "node_in_dim": 44, 
    "edge_in_dim": 12, 
    "num_global_feats": 1, 
    "hidden_dim": hidden_dim,
    "mlp_out_hidden": mlp, 
    "num_layers_atom": atom_layers, 
    "num_layers_mol": mol_layers,
    "L1_layers_atom": 4,
    "L1_layers_mol": 1,
    "L1_dropout": 0.142,
    "L2_layers_atom": 2,
    "L2_layers_mol": 3,
    "L2_dropout": 0.255,
    "L3_layers_atom": 1,
    "L3_layers_mol": 3,
    "L3_dropout": 0.026,
    "L1_hidden_dim": 247,
    "L2_hidden_dim": 141,
    "L3_hidden_dim": 180,
}

# Initialize Model
model = GroupGAT_jittable.GCGAT_v4pro_jit(net_params).to(device)

# Optimizer, Scheduler, and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, min_lr=learning_rate, patience=30
)
loss_func = torch.nn.functional.mse_loss

# Early Stopping
early_stopper = EarlyStopping(patience=patience, model_name='gcgat_model', skip_save=True)

# Model Checkpoint
model_filename = 'gcgat_latest_pdd.pth'

if os.path.exists(model_filename):
    print(f"Loading model from '{model_filename}'.")
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
else:
    print("Training the model...")
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
        early_stopper=early_stopper,
        model_needs_frag=True,
        net_params=net_params
    )
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to '{model_filename}'.")

# Evaluation Function
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

# Generate Predictions
train_preds, train_targets = test_model_jit_with_parity(model, train, device, batch_size, model_needs_frag=True)
val_preds, val_targets = test_model_jit_with_parity(model, val, device, batch_size, model_needs_frag=True)
test_preds, test_targets = test_model_jit_with_parity(model, test, device, batch_size, model_needs_frag=True)

# Compute Metrics
train_metrics = pred_metric(prediction=train_preds, target=train_targets, metrics='all', print_out=True)
val_metrics = pred_metric(prediction=val_preds, target=val_targets, metrics='all', print_out=True)
test_metrics = pred_metric(prediction=test_preds, target=test_targets, metrics='all', print_out=True)

# Example: Access Specific Metrics
train_mae = train_metrics.get('mae')
val_mae = val_metrics.get('mae')
test_mae = test_metrics.get('mae')
print(f"Train MAE: {train_mae:.3f}, Validation MAE: {val_mae:.3f}, Test MAE: {test_mae:.3f}")

# Parity Plot (Optional)
def create_parity_plot(train_preds, train_targets, val_preds, val_targets, test_preds, test_targets):
    import matplotlib.pyplot as plt

    # Convert tensors to numpy arrays
    train_preds_np = train_preds.numpy()
    train_targets_np = train_targets.numpy()
    val_preds_np = val_preds.numpy()
    val_targets_np = val_targets.numpy()
    test_preds_np = test_preds.numpy()
    test_targets_np = test_targets.numpy()

    # Concatenate data
    all_preds = np.concatenate([train_preds_np, val_preds_np, test_preds_np])
    all_targets = np.concatenate([train_targets_np, val_targets_np, test_targets_np])
    all_labels = np.concatenate([
        np.full(len(train_preds_np), 'Train'),
        np.full(len(val_preds_np), 'Validation'),
        np.full(len(test_preds_np), 'Test')
    ])

    # Color Mapping
    colors = {'Train': 'blue', 'Validation': 'green', 'Test': 'red'}
    color_list = [colors[label] for label in all_labels]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, c=color_list, alpha=0.6, label=all_labels)
    min_val, max_val = all_targets.min(), all_targets.max()
    buffer = (max_val - min_val) * 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlim([min_val - buffer, max_val + buffer])
    plt.ylim([min_val - buffer, max_val + buffer])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Parity Plot')
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
    ])
    plt.show()

# Uncomment to create Parity Plot
#create_parity_plot(train_preds, train_targets, val_preds, val_targets, test_preds, test_targets)
