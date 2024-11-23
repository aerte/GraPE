from grape_chem.models import GroupGAT
from grape_chem.utils import (
    DataSet, train_model, EarlyStopping, split_data, 
    test_model, pred_metric, return_hidden_layers, set_seed, JT_SubGraph
)
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
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

# --------------------- Hyperparameters ---------------------
epochs = 1000
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.00126
weight_decay = 1e-4
mlp_layers = 2
atom_layers = 3
mol_layers = 3
# ------------------------------------------------------------

# --------------------- Data Loading -------------------------
root = './env/ICP.xlsx'
# root = './env/Solvation__splits.csv'
# in solvation the global feat is ['Temperature'], target is ['Energy'], subset is ['Split']
sheet_name = ''

df = pd.read_excel(root)  # .iloc[:25]
# df = pd.read_csv(root)
smiles = df['SMILES'].to_numpy()
target = df['Value'].to_numpy()

# Specific to one xlsx with a "Tag" column
tags = df['Subset'].to_numpy()
unique_tags = np.unique(tags)
tag_to_int = {'Training': 0, 'Validation': 1, 'Test': 2}
# tag_to_int = {'train': 0, 'val': 1, 'test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

### Global feature from sheet, uncomment if necessary
# global_feats = df['Global Feats'].to_numpy()

# Global features (e.g., temperature)
global_feats = df['T'].to_numpy()
# Standardize before loading into DataSet
mean_target, std_target = np.mean(target), np.std(target)
target = standardize(target, mean_target, std_target)
mean_global_feats, std_global_feats = np.mean(global_feats), np.std(global_feats)
global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

# -------------------- Fragmentation -------------------------
fragmentation_scheme = "MG_plus_reference"
print("Initializing fragmentation...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path="env/default_Vc_cace_frag")
frag_dim = fragmentation.frag_dim
print("Fragmentation initialized.")
# ------------------------------------------------------------

# --------------------- Create Dataset ------------------------
data = DataSet(
    smiles=smiles, 
    target=target, 
    global_features=global_feats,  # Pass the standardized global features
    filter=True, 
    fragmentation=fragmentation
)
# ------------------------------------------------------------

# ---------------------- Split Data --------------------------
train, val, test = split_data(data, split_type='custom', custom_split=custom_split)
# ------------------------------------------------------------

# -------------------- Device Configuration -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------------------------------------

# -------------------- Model Parameters ------------------------
mlp = return_hidden_layers(mlp_layers)
net_params = {
    "device": device,
    "num_atom_type": 44,         # Ensure this matches your featurizer
    "num_bond_type": 12,         # Ensure this matches your featurizer
    "dropout": 0.0,
    "MLP_layers": mlp_layers,
    "frag_dim": frag_dim,
    "final_dropout": 0.119,
    "use_global_features": True,  # Enable use of global features
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
    "L3_layers_mol": 4,
    "L3_dropout": 0.026,
    "L1_hidden_dim": 247,
    "L2_hidden_dim": 141,
    "L3_hidden_dim": 47,
}

# Initialize Model
model = GroupGAT.GCGAT_v4pro(net_params).to(device)

# ----------------- Optimizer, Scheduler, Loss ----------------
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.9, min_lr=1e-9, patience=patience
)
loss_func = torch.nn.functional.l1_loss

# Early Stopping
early_stopper = EarlyStopping(patience=patience, model_name='gcgat_model', skip_save=True)

# Model Checkpoint
model_filename = 'gcgat_latest.pth'

# ------------------- Load or Train Model ---------------------
if os.path.exists(model_filename):
    print(f"Loading model from '{model_filename}'.")
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
else:
    print("Training the model...")
    train_model(
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
        early_stopper=early_stopper
    )
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to '{model_filename}'.")
# ------------------------------------------------------------

# ------------------- Evaluation Function ---------------------
def test_model_with_parity(
    model: torch.nn.Module, 
    test_data_loader: Union[List[Data], DataLoader],
    device: torch.device,
    batch_size: int = 32, 
    model_needs_frag: bool = False
) -> Tuple[Tensor, Tensor]:
    model.eval()
    test_loader = test_data_loader if isinstance(test_data_loader, DataLoader) else DataLoader(test_data_loader, batch_size=batch_size)
    preds, targets = [], []
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Evaluating") as pbar:
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch) if model_needs_frag else model(batch)
                preds.append(out.detach().cpu())
                targets.append(batch.y.detach().cpu())
                pbar.update(1)
    
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return preds, targets
# ------------------------------------------------------------

# ------------------ Generate Predictions ---------------------
train_preds, train_targets = test_model_with_parity(model, train, device, batch_size, model_needs_frag=True)
val_preds, val_targets = test_model_with_parity(model, val, device, batch_size, model_needs_frag=True)
test_preds, test_targets = test_model_with_parity(model, test, device, batch_size, model_needs_frag=True)
# ------------------------------------------------------------

# --------------------- Compute Metrics ------------------------
train_metrics = pred_metric(prediction=train_preds, target=train_targets, metrics='all', print_out=True)
val_metrics = pred_metric(prediction=val_preds, target=val_targets, metrics='all', print_out=True)
test_metrics = pred_metric(prediction=test_preds, target=test_targets, metrics='all', print_out=True)

# Example: Access Specific Metrics
train_mae = train_metrics.get('mae')
val_mae = val_metrics.get('mae')
test_mae = test_metrics.get('mae')
print(f"Train MAE: {train_mae:.3f}, Validation MAE: {val_mae:.3f}, Test MAE: {test_mae:.3f}")
# ------------------------------------------------------------

# --------------------- Parity Plot (Optional) -----------------
def create_parity_plot(train_preds, train_targets, val_preds, val_targets, test_preds, test_targets):
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
    plt.scatter(all_targets, all_preds, c=color_list, alpha=0.6)
    
    # Plot y=x line
    min_val, max_val = all_targets.min(), all_targets.max()
    buffer = (max_val - min_val) * 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    # Set limits with buffer
    plt.xlim([min_val - buffer, max_val + buffer])
    plt.ylim([min_val - buffer, max_val + buffer])
    
    # Labels and title
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Parity Plot')
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
    ])
    plt.show()

# Uncomment the line below to generate the Parity Plot
# create_parity_plot(train_preds, train_targets, val_preds, val_targets, test_preds, test_targets)
# ------------------------------------------------------------
