# Import necessary libraries
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
from torch_geometric.data import DataLoader, Batch
import os

# Import the modified model
from grape_chem.models import GroupGAT_ICP  # Ensure this file contains the above model code

def standardize(x, mean, std):
    return (x - mean) / std

##########################################################################################
#####################    Data Input Region  ##############################################
##########################################################################################

set_seed(14102024)

# Hyperparameters
epochs = 12000
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.005
weight_decay = 1e-6
mlp_layers = 2
atom_layers = 3
mol_layers = 3

# Specify the data file
root = './env/ICP.xlsx'

df = pd.read_excel(root)
smiles = df['SMILES'].to_numpy()
target = df['Value'].to_numpy()

# Handle custom splits
tags = df['Subset'].to_numpy()
tag_to_int = {'Training': 0, 'Validation': 1, 'Test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

# Global feature T
global_feats = df['T'].to_numpy()

# Standardize targets and global features
mean_target, std_target = np.mean(target), np.std(target)
#target = standardize(target, mean_target, std_target)
mean_global_feats, std_global_feats = np.mean(global_feats), np.std(global_feats)
#global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

# Fragmentation
fragmentation_scheme = "MG_plus_reference"
fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path='env/icp_single_task_fragmentation')
frag_dim = fragmentation.frag_dim

# DataSet Creation
data = DataSet(
    smiles=smiles,
    target=target,
    global_features=global_feats,
    filter=True,
    fragmentation=fragmentation,
)

# Split data using custom splits
train, val, test = split_data(data, split_type='custom', custom_split=custom_split)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define network parameters
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
    "L3_layers_mol": 4,
    "L3_dropout": 0.026,
    "L1_hidden_dim": 247,
    "L2_hidden_dim": 141,
    "L3_hidden_dim": 47,
}

# Initialize the model
model = GroupGAT_ICP(net_params)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_Stopper = EarlyStopping(patience=50, model_name='groupgat_icp', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, min_lr=0.005, patience=30
)

# Use MSE loss
loss_func = torch.nn.functional.mse_loss

# Define model filename
model_filename = 'groupgat_icp_coupled_multitask.pth'

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
pred = test_model_jit(
    model=model,
    test_data_loader=test,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)

# Rescale predictions and targets
pred_rescaled = pred * std_target + mean_target
test_target_rescaled = test.y * std_target + mean_target

# Compute metrics for Test set
print("\n--- Test Set Metrics ---")
test_metrics = pred_metric(prediction=pred_rescaled, target=test_target_rescaled, metrics='all', print_out=True)

####### Example for overall evaluation of the MAE #########
train_preds = test_model_jit(
    model=model,
    test_data_loader=train,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)
val_preds = test_model_jit(
    model=model,
    test_data_loader=val,
    device=device,
    batch_size=batch_size,
    model_needs_frag=True
)

# Rescale predictions and targets
train_preds_rescaled = train_preds * std_target + mean_target
train_targets_rescaled = train.y * std_target + mean_target

val_preds_rescaled = val_preds * std_target + mean_target
val_targets_rescaled = val.y * std_target + mean_target

# Calculate and print metrics for Validation set
print("\n--- Validation Set Metrics ---")
val_metrics = pred_metric(prediction=val_preds_rescaled, target=val_targets_rescaled, metrics='all', print_out=True)

# Calculate and print metrics for Overall (Entire Data)
# Concatenate all predictions and targets
overall_preds_rescaled = torch.cat([train_preds_rescaled, val_preds_rescaled, pred_rescaled], dim=0)

# we need to convert to torch tensors:
if isinstance(train_targets_rescaled, np.ndarray):
    train_targets_rescaled = torch.from_numpy(train_targets_rescaled).float().to(device)

if isinstance(val_targets_rescaled, np.ndarray):
    val_targets_rescaled = torch.from_numpy(val_targets_rescaled).float().to(device)

if isinstance(test_target_rescaled, np.ndarray):
    test_target_rescaled = torch.from_numpy(test_target_rescaled).float().to(device)

overall_targets_rescaled = torch.cat([train_targets_rescaled, val_targets_rescaled, test_target_rescaled], dim=0)

print("\n--- Overall (Entire Data) Metrics ---")
overall_metrics = pred_metric(prediction=overall_preds_rescaled, target=overall_targets_rescaled, metrics='all', print_out=True)

####### Creating Parity Plot #########
def create_parity_plot(
    train_preds_rescaled, train_targets_rescaled,
    val_preds_rescaled, val_targets_rescaled,
    test_preds_rescaled, test_targets_rescaled,
):
    import matplotlib.pyplot as plt

    val_targets_rescaled = val_targets_rescaled.cpu().numpy()
    test_targets_rescaled = test_targets_rescaled.cpu().numpy()
    train_targets_rescaled = train_targets_rescaled.cpu().numpy()

    # Combine all datasets
    all_preds = np.concatenate([
        train_preds_rescaled,
        val_preds_rescaled,
        test_preds_rescaled
    ])
    all_targets = np.concatenate([
        train_targets_rescaled,
        val_targets_rescaled,
        test_targets_rescaled
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
    plt.xlabel('Actual Cp (kJ/kmol⋅K')
    plt.ylabel('Predicted Cp (kJ/kmol⋅K')
    plt.title('Parity Plot for Cp Prediction - Coupled Multitask (GroupGAT)')
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
    ])
    plt.show()

# Generate parity plot
create_parity_plot(
    train_preds_rescaled, train_targets_rescaled,
    val_preds_rescaled, val_targets_rescaled,
    pred_rescaled, test_target_rescaled,
)

####### Function to Generate Predictions and Save to CSV #########
def generate_predictions_and_save(
    data,
    model,
    df,
    smiles,
    target,
    batch_size,
    device,
    output_filename
):
    """
    Function to generate predictions for the full dataset, align them with the SMILES, and save to a CSV file.
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

    # Step 2: Filter df
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)

    # Step 3: Ensure lengths match
    if len(df_filtered) != len(data):
        print(f"Length mismatch: df_filtered ({len(df_filtered)}) vs. data ({len(data)})")
        # Attempt to extract SMILES from data
        smiles_list = []
        for d in data:
            if hasattr(d, 'smiles'):
                smiles_list.append(d.smiles)
            else:
                smiles_list.append(None)  # Placeholder if SMILES not available
        if len(smiles_list) == len(data):
            print("Extracted SMILES from data objects.")
            df_filtered = df_filtered.iloc[:len(data)].reset_index(drop=True)
            df_filtered['SMILES'] = smiles_list
        else:
            print("Cannot retrieve SMILES from data. Exiting.")
            return
    else:
        print(f"Lengths match: {len(df_filtered)} entries")

    # Step 4: Generate predictions
    print("Generating predictions for the entire dataset...")
    all_preds = test_model_jit(
        model=model,
        test_data_loader=data,
        device=device,
        batch_size=batch_size,
        model_needs_frag=True
    )

    # Step 5: No need to rescale predictions, since targets were not standardized

    # Step 6: Create DataFrame
    predictions_np = all_preds.cpu().numpy().flatten()
    df_pred = df_filtered.copy()
    df_pred['Predicted_Value'] = predictions_np

    # Step 7: Save to CSV
    df_pred.to_csv(output_filename, index=False)
    print(f"Predictions saved to '{output_filename}'.")

# Call the function to generate predictions and save to CSV
generate_predictions_and_save(
    data=data,
    model=model,
    df=df,
    smiles=smiles,
    target=target,
    batch_size=batch_size,
    device=device,
    output_filename='predictions_coupled_multitask.csv'
)
