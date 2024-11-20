import torch
import torch.nn as nn
from grape_chem.models import GroupGAT_Ensemble
from grape_chem.utils import (
    DataSet, EarlyStopping, train_model, test_model,
    set_seed, JT_SubGraph, pred_metric
)
from torch_geometric.loader import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import os

# Set seed for reproducibility
set_seed(42)

def standardize(x, mean, std):
    return (x - mean) / std

def split_data_by_indices(data, train_idx, val_idx, test_idx):
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]
    return train, val, test

def train_and_evaluate_model(
    target_name, smiles, target_values, fragmentation,
    custom_split_indices, net_params, device, batch_size, epochs,
    learning_rate, weight_decay, scheduler_patience, global_feats=None
):
    # Standardize target
    mean_target = np.mean(target_values)
    std_target = np.std(target_values)
    target_standardized = standardize(target_values, mean_target, std_target)
    if global_feats is not None:
        global_feats = standardize(global_feats, np.mean(global_feats), np.std(global_feats))
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

    # Initialize the ensemble model
    num_models = 5  # Number of models in the ensemble
    model = GroupGAT_Ensemble(net_params, num_models).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, min_lr=1.00E-09, patience=scheduler_patience
    )
    early_Stopper = EarlyStopping(patience=patience, model_name='random', skip_save=True)
    loss_func = torch.nn.functional.mse_loss

    # Define model filename
    model_filename = f'gcgat_coupled_ensemble_model_{target_name}.pth'

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

    print(f"\nMetrics for property '{target_name}' on training set:")
    train_metrics = pred_metric(
        prediction=train_pred_rescaled,
        target=train_target_rescaled,
        metrics='all',
        print_out=True
    )

    # Calculate and print metrics for validation set
    print(f"\nMetrics for property '{target_name}' on validation set:")
    val_metrics = pred_metric(
        prediction=val_pred_rescaled,
        target=val_target_rescaled,
        metrics='all',
        print_out=True
    )
    
    # Calculate and print metrics for test set
    print(f"\nMetrics for property '{target_name}' on test set:")
    test_metrics = pred_metric(
        prediction=test_pred_rescaled,
        target=test_target_rescaled,
        metrics='all',
        print_out=True
    )

    return results

def load_icp_dataset(fragmentation, pass_on_data=False):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.getcwd())
    # Construct the absolute path to ICP.xlsx
    root = '/home/paul/Documents/spaghetti/GraPE/env/ICP.xlsx'
    print(f"Loading data from: {root}")
    df = pd.read_excel(root)

    smiles = df['SMILES'].to_numpy()
    targets = df['Value'].to_numpy()

    tags = df['Subset'].to_numpy()
    tag_to_int = {'Training': 0, 'Validation': 1, 'Test': 2}
    custom_split = np.array([tag_to_int[tag] for tag in tags])

    global_feats = df['T'].to_numpy()

    train_indices = np.where(custom_split == 0)[0]
    val_indices = np.where(custom_split == 1)[0]
    test_indices = np.where(custom_split == 2)[0]

    # Standardize target
    mean_target = np.mean(targets)
    std_target = np.std(targets)
    target_standardized = standardize(targets, mean_target, std_target)
    global_feats_standardized = standardize(global_feats, np.mean(global_feats), np.std(global_feats))

    # Create DataSet
    data = DataSet(
        smiles=smiles,
        target=target_standardized,
        global_features=global_feats_standardized,
        filter=True,
        fragmentation=fragmentation
    )

    # Split data
    train_data, val_data, test_data = split_data_by_indices(
        data, train_indices, val_indices, test_indices
    )
    if pass_on_data:
        return train_data, val_data, test_data, mean_target, std_target, data, df, smiles
    return train_data, val_data, test_data, mean_target, std_target

def return_hidden_layers(num):
    return [2 ** i * 32 for i in range(num, 0, -1)]

def get_net_params_per_model(config, device, frag_dim):
    net_params_per_model = {}
    model_names = ['A', 'B', 'C', 'D', 'E']
    for model in model_names:
        net_params = {
            "device": device,
            "num_atom_type": 44,
            "num_bond_type": 12,
            "dropout": config[f"{model}_dropout"],
            "MLP_layers": int(config[f"{model}_MLP_layers"]),
            "frag_dim": frag_dim,
            "final_dropout": config[f"{model}_final_dropout"],
            "num_heads": 1,  # If you wish, you can make this a hyperparameter
            "node_in_dim": 44,
            "edge_in_dim": 12,
            "num_global_feats": 1,
            "hidden_dim": int(config[f"{model}_hidden_dim"]),
            "mlp_out_hidden": return_hidden_layers(int(config[f"{model}_MLP_layers"])),
            "num_layers_atom": int(config[f"{model}_num_layers_atom"]),
            "num_layers_mol": int(config[f"{model}_num_layers_mol"]),
            # L1 parameters
            "L1_layers_atom": int(config[f"{model}_L1_layers_atom"]),
            "L1_layers_mol": int(config[f"{model}_L1_layers_mol"]),
            "L1_dropout": config[f"{model}_L1_dropout"],
            "L1_hidden_dim": int(config[f"{model}_L1_hidden_dim"]),
            # L2 parameters
            "L2_layers_atom": int(config[f"{model}_L2_layers_atom"]),
            "L2_layers_mol": int(config[f"{model}_L2_layers_mol"]),
            "L2_dropout": config[f"{model}_L2_dropout"],
            "L2_hidden_dim": int(config[f"{model}_L2_hidden_dim"]),
            # L3 parameters
            "L3_layers_atom": int(config[f"{model}_L3_layers_atom"]),
            "L3_layers_mol": int(config[f"{model}_L3_layers_mol"]),
            "L3_dropout": config[f"{model}_L3_dropout"],
            "L3_hidden_dim": int(config[f"{model}_L3_hidden_dim"]),
            "output_dim": 1,
            #disable global feats for the individual models
            "use_global_features": False,
            "num_global_feats": 0,
        }
        net_params_per_model[model] = net_params
    return net_params_per_model

def load_model(config, device, frag_dim):
    net_params_per_model = get_net_params_per_model(config, device, frag_dim)
    return GroupGAT_Ensemble(net_params_per_model).to(device)

def generate_predictions_and_save(
    data,
    model,
    df,
    smiles,
    mean_target,
    std_target,
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
    with torch.no_grad():
        all_preds = test_model(
            model=model,
            test_data_loader=data,
            device=device,
            batch_size=batch_size
        )

    # Step 5: Rescale predictions (since targets were standardized)
    all_preds_rescaled = all_preds * std_target + mean_target

    # Step 6: Create DataFrame
    predictions_np = all_preds_rescaled.cpu().numpy().flatten()  # No need to detach since gradients are not tracked
    df_pred = df_filtered.copy()
    df_pred['Predicted_Value'] = predictions_np

    # Step 7: Save to CSV
    df_pred.to_csv(output_filename, index=False)
    print(f"Predictions saved to '{output_filename}'.")

def create_parity_plot(
    train_preds_rescaled, train_targets_rescaled,
    val_preds_rescaled, val_targets_rescaled,
    test_preds_rescaled, test_targets_rescaled,
    property_name='Property'
):
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert tensors to numpy arrays
    train_preds_rescaled = train_preds_rescaled.cpu().numpy()
    val_preds_rescaled = val_preds_rescaled.cpu().numpy()
    test_preds_rescaled = test_preds_rescaled.cpu().numpy()
    train_targets_rescaled = train_targets_rescaled.cpu().numpy()
    val_targets_rescaled = val_targets_rescaled.cpu().numpy()
    test_targets_rescaled = test_targets_rescaled.cpu().numpy()

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
    plt.xlabel(f'Actual {property_name}')
    plt.ylabel(f'Predicted {property_name}')
    plt.title(f'Parity Plot for {property_name} Prediction')
    plt.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Test', markerfacecolor='red', markersize=10)
    ])
    plt.show()

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize fragmentation
    fragmentation_scheme = "MG_plus_reference"
    print("Initializing fragmentation...")
    fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path='/home/paul/Documents/spaghetti/GraPE/env/ICP_fragmentation.pth')
    print("Done.")

    # Load the ICP dataset
    train_data, val_data, test_data, mean_target, std_target, data, df, smiles = load_icp_dataset(fragmentation, pass_on_data=True)

    # Prepare the configuration dictionary
    config = {
        'initial_lr': 0.001,
        'weight_decay': 1e-5,
        'lr_reduction_factor': 0.5,
        'batch_size': 32,
    }

    model_names = ['A', 'B', 'C', 'D', 'E']
    for model in model_names:
        config[f"{model}_dropout"] = 0.1
        config[f"{model}_MLP_layers"] = 2
        config[f"{model}_final_dropout"] = 0.1
        config[f"{model}_hidden_dim"] = 128
        config[f"{model}_num_layers_atom"] = 3
        config[f"{model}_num_layers_mol"] = 3
        config[f"{model}_L1_layers_atom"] = 2
        config[f"{model}_L1_layers_mol"] = 2
        config[f"{model}_L1_dropout"] = 0.1
        config[f"{model}_L1_hidden_dim"] = 64
        config[f"{model}_L2_layers_atom"] = 2
        config[f"{model}_L2_layers_mol"] = 2
        config[f"{model}_L2_dropout"] = 0.1
        config[f"{model}_L2_hidden_dim"] = 64
        config[f"{model}_L3_layers_atom"] = 2
        config[f"{model}_L3_layers_mol"] = 2
        config[f"{model}_L3_dropout"] = 0.1
        config[f"{model}_L3_hidden_dim"] = 64

    # Load the model
    frag_dim = fragmentation.frag_dim
    model = load_model(config, device, frag_dim)
    model.to(device)

    # Set up optimizer, scheduler, and early stopper
    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    early_stopper = EarlyStopping(patience=30, model_name='GroupGAT_Ensemble', skip_save=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'], min_lr=1e-10, patience=10)
    loss_function = torch.nn.functional.mse_loss

    # Create data loaders
    batch_size = config.get('batch_size', len(train_data))  # Adjust batch size if necessary
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    iterations = 3000  # Adjust as necessary
    start_epoch = 0

    generate_predictions_and_save(
        data=data,
        model=model,
        df=df,
        smiles=smiles,
        mean_target=mean_target,
        std_target=std_target,
        batch_size=batch_size,
        device=device,
        output_filename='predictions_groupGAT_coupled_singletask.csv'
    )

    model.eval()
    with torch.no_grad():
        # Generate predictions on train, validation, and test sets
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
        test_pred = test_model(
            model=model,
            test_data_loader=test_data,
            device=device,
            batch_size=batch_size
        )

    # Collect targets
    train_target = torch.tensor([data.y for data in train_data]).to(device)
    val_target = torch.tensor([data.y for data in val_data]).to(device)
    test_target = torch.tensor([data.y for data in test_data]).to(device)

    # Rescale predictions and targets
    train_pred_rescaled = train_pred * std_target + mean_target
    val_pred_rescaled = val_pred * std_target + mean_target
    test_pred_rescaled = test_pred * std_target + mean_target

    train_target_rescaled = train_target * std_target + mean_target
    val_target_rescaled = val_target * std_target + mean_target
    test_target_rescaled = test_target * std_target + mean_target

    # Calculate and print metrics
    print(f"\nMetrics on Training Set:")
    train_metrics = pred_metric(
        prediction=train_pred_rescaled,
        target=train_target_rescaled,
        metrics='all',
        print_out=True
    )

    print(f"\nMetrics on Validation Set:")
    val_metrics = pred_metric(
        prediction=val_pred_rescaled,
        target=val_target_rescaled,
        metrics='all',
        print_out=True
    )

    print(f"\nMetrics on Test Set:")
    test_metrics = pred_metric(
        prediction=test_pred_rescaled,
        target=test_target_rescaled,
        metrics='all',
        print_out=True
    )

    # Overall MAE across all datasets
    overall_mae = (train_metrics['mae'] + val_metrics['mae'] + test_metrics['mae']) / 3
    print(f'\nOverall MAE across all datasets: {overall_mae:.4f}')

    # Generate parity plot
    create_parity_plot(
        train_pred_rescaled, train_target_rescaled,
        val_pred_rescaled, val_target_rescaled,
        test_pred_rescaled, test_target_rescaled,
        property_name='ICP'  # Replace with the appropriate property name
    )

    # Evaluate on test set
    #model.eval()
    #test_loss = val_epoch(model=model, loss_func=loss_function, val_loader=test_loader, device=device)
    #print(f"Test Loss: {test_loss:.4f}")