import copy
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm

from grape_chem.models import GroupGAT_jittable
from grape_chem.utils import DataSet, train_model_jit, EarlyStopping, split_data, test_model_jit, pred_metric, return_hidden_layers, set_seed, JT_SubGraph, FragmentGraphDataSet
from grape_chem.datasets import FreeSolv 
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

epochs = 1000
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.01054627
weight_decay = 1e-6
mlp_layers = 2
atom_layers = 3
mol_layers = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            early_stopper=early_stopper,
            model_needs_frag=True,
            net_params=net_params
        )

        # Evaluate the model
        val_preds = test_model_jit(
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

# Now you can call this function with your data loaders and parameters
results = transfer_learning_loop(
    base_model=model,
    train_loader=train,
    val_loader=val,
    num_iterations=5,
    epochs=10,
    learning_rate=0.001,
    weight_decay=1e-6,
    patience=5
)