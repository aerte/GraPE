# Import necessary modules and functions
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch
from grape_chem.models import AGC  # Import AGC model
# from grape_chem.models import GroupGAT_jittable  # No longer needed
from grape_chem.utils import (
    DataSet,
    train_model,
    EarlyStopping,
    split_data,
    test_model,
    pred_metric,
    return_hidden_layers,
    set_seed,
    JT_SubGraph,
    FragmentGraphDataSet,
)
from grape_chem.datasets import FreeSolv
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from tqdm import tqdm
from typing import Union, List, Tuple
from torch import Tensor
import os
import copy

# Define helper functions
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
learning_rate = 0.01054627
weight_decay = 1e-6
mlp_layers = 2
atom_layers = 3
mol_layers = 3

# Change to your own specifications
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

########################## Fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("Initializing fragmentation...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path="env/default_ICP_frag")
frag_dim = fragmentation.frag_dim
print("Done.")

########################### Data Preparation #####################################
# Example with FreeSolv dataset
# data = FreeSolv(fragmentation=fragmentation)

# Using custom dataset (e.g., QM9, testing, or Excel data)
data = DataSet(
    smiles=smiles,
    target=target,
    global_features=global_feats,
    filter=True,
    fragmentation=fragmentation,
)

# Split data
train, val, test = split_data(
    data, split_type='custom', custom_split=custom_split,
)

##################################################################################
# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define network parameters
mlp = return_hidden_layers(mlp_layers)
net_params = {
    "device": device,  # Shouldn't be passed in this way, but best we have for now
    "num_atom_type": 44,  # == node_in_dim
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
}

# Create the AGC model instance
model = AGC(net_params)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_Stopper = EarlyStopping(patience=50, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, min_lr=1.00E-09, patience=15
)

loss_func = torch.nn.functional.mse_loss

model.to(device)

# Define model filename
model_filename = 'agc_latest.pth'

# Check if the model file exists
if os.path.exists(model_filename):
    print(f"Model file '{model_filename}' found. Loading the trained model.")
    # Load the model state dict
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
else:
    print(f"No trained model found at '{model_filename}'. Proceeding to train the model.")
    # Train the model
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
        early_stopper=early_Stopper,
        model_needs_frag=True,
        model_name=model_filename,
    )
    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to '{model_filename}'.")

####### Generating prediction tensor for the TEST set (Not rescaled) #########

pred = test_model(
    model=model,
    test_data_loader=test,
    device=device,
    batch_size=batch_size,
)
pred_metric(prediction=pred, target=test.y, metrics='all', print_out=True)

# ---------------------------------------------------------------------------------------

####### Example for rescaling the MAE prediction ##########
if False:
    test_mae = pred_metric(prediction=pred, target=test.y, metrics='mae', print_out=False)['mae']
    # test_mae_rescaled = test_mae * std_target + mean_target  # TODO: Add rescaling if necessary
    # print(f'Rescaled MAE for the test set {test_mae_rescaled:.3f}')

    # ---------------------------------------------------------------------------------------

    ####### Example for overall evaluation of the MAE #########

    train_preds = test_model(
        model=model,
        test_data_loader=train,
        device=device,
    )
    val_preds = test_model(
        model=model,
        test_data_loader=val,
        device=device,
    )

    train_mae = pred_metric(prediction=train_preds, target=train.y, metrics='mae', print_out=False)['mae']
    val_mae = pred_metric(prediction=val_preds, target=val.y, metrics='mae', print_out=False)['mae']

    # overall_mae = (train_mae + val_mae + test_mae) / 3 * std_target + mean_target
    # print(f'Rescaled overall MAE {overall_mae:.3f}')

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

    # Function to perform transfer learning on the model
    def transfer_learning_loop(
        base_model,
        train_loader,
        val_loader,
        num_iterations=5,
        epochs=10,
        learning_rate=0.001,
        weight_decay=1e-6,
        patience=5
    ):
        # Initialize a list to store results for each iteration
        results = []

        # Loop for multiple transfer learning iterations
        for i in range(num_iterations):
            print(f"\nTransfer Learning Iteration {i + 1}/{num_iterations}")

            # Clone the base model for each transfer learning iteration
            model = copy.deepcopy(base_model)
            model.to(device)

            # Freeze all layers except the last two
            freeze_model_layers(model)

            # Define a new optimizer for the un-frozen parameters
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay
            )

            # Set up early stopping for each iteration
            early_stopper = EarlyStopping(
                patience=patience,
                model_name=f"transfer_learning_model_{i}",
                skip_save=True
            )

            # Train the model for the specified number of epochs
            train_model(
                model=model,
                loss_func=loss_func,
                optimizer=optimizer,
                train_data_loader=train_loader,
                val_data_loader=val_loader,
                epochs=epochs,
                device=device,
                batch_size=batch_size,
                scheduler=scheduler,
                early_stopper=early_stopper,
                model_needs_frag=True,
                model_name=None
            )

            # Evaluate the model
            val_preds = test_model(
                model=model,
                test_data_loader=val_loader,
                device=device,
                batch_size=batch_size,
                model_needs_frag=True
            )
            val_mae = pred_metric(
                prediction=val_preds,
                target=torch.cat([data.y for data in val_loader], dim=0),
                metrics='mae',
                print_out=False
            )['mae']
            print(f"Validation MAE after transfer learning iteration {i+1}: {val_mae}")

            # Store results
            results.append({
                'iteration': i + 1,
                'model': model,
                'val_mae': val_mae
            })

        return results

    results = transfer_learning_loop(
        base_model=model,
        train_loader=train,
        val_loader=val,
        num_iterations=5,
        epochs=300,
        learning_rate=0.001,
        weight_decay=1e-6,
        patience=5
    )

    # Modify the test_model function to return both predictions and targets
    def test_model_with_parity(
        model: torch.nn.Module,
        test_data_loader: Union[List, DataLoader],
        device: str = None,
        batch_size: int = 32,
        return_latents: bool = False,
        model_needs_frag: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Auxiliary function to test a trained model and return the predictions as well as the targets and optional latent node
        representations. Can initialize DataLoaders if only list of Data objects are given.

        Parameters
        ----------
        model : torch.nn.Module
            Model that will be tested. Has to be a torch Module.
        test_data_loader : list of Data or DataLoader
            A list of Data objects or the DataLoader directly to be used as the test graphs.
        device : str
            Torch device to be used ('cpu', 'cuda', or 'mps'). Default is 'cpu'.
        batch_size : int
            Batch size of the DataLoader if not given directly. Default is 32.
        return_latents : bool
            Determines if the latents should be returned. **If used, the model must include return_latent statement**.
            Default is False.
        model_needs_frag : bool
            Indicates whether the model requires fragment information.

        Returns
        -------
        Tuple[Tensor, Tensor] or Tuple[Tensor, Tensor, Tensor]
            Returns predictions and targets, and optionally latents if return_latents is True.
        """

        device = torch.device('cpu') if device is None else device

        if not isinstance(test_data_loader, DataLoader):
            test_data_loader = DataLoader(
                [data for data in test_data_loader], batch_size=batch_size
            )

        model.eval()

        with tqdm(total=len(test_data_loader)) as pbar:

            for idx, batch in enumerate(test_data_loader):
                batch = batch.to(device)
                # Handling models that require fragment information
                if model_needs_frag:
                    if return_latents:
                        out, lat = model(batch, return_lats=True)
                    else:
                        out = model(batch)
                else:
                    if return_latents:
                        out, lat = model(batch, return_lats=True)
                    else:
                        out = model(batch)

                # Collect targets
                target = batch.y

                # Concatenate predictions, targets, and latents
                if idx == 0:
                    preds = out
                    targets = target
                    if return_latents:
                        latents = lat
                else:
                    preds = torch.cat([preds, out], dim=0)
                    targets = torch.cat([targets, target], dim=0)
                    if return_latents:
                        latents = torch.cat([latents, lat], dim=0)

                pbar.update(1)

        if return_latents:
            return preds, targets, latents
        return preds, targets

    ####### Generating predictions and targets for all datasets #########

    train_preds, train_targets = test_model_with_parity(
        model=model,
        test_data_loader=train,
        device=device,
        batch_size=batch_size,
        model_needs_frag=True
    )

    val_preds, val_targets = test_model_with_parity(
        model=model,
        test_data_loader=val,
        device=device,
        batch_size=batch_size,
        model_needs_frag=True
    )

    test_preds, test_targets = test_model_with_parity(
        model=model,
        test_data_loader=test,
        device=device,
        batch_size=batch_size,
        model_needs_frag=True
    )
