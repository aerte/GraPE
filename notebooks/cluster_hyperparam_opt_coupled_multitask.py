import torch
import torch.nn as nn
from grape_chem.models import GroupGAT_ICP
from grape_chem.utils import (
    DataSet, EarlyStopping, set_seed, JT_SubGraph, 
    return_hidden_layers, train_epoch_jittable, val_epoch_jittable
)
from torch_geometric.loader import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import os
import tempfile
from functools import partial

import ray
from ray import tune, train
from ray.air import RunConfig
from ray.train import Checkpoint
from ray.tune import Tuner
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
import ConfigSpace as CS
import argparse
import yaml
from jinja2 import Template

# Initialize Ray
ray.init()

# Set seed for reproducibility
set_seed(42)

def standardize(x, mean, std):
    return (x - mean) / std

def load_icp_dataset(fragmentation, data_config):
    # Read parameters from data_config
    file_name = data_config.get('file_name')
    file_type = data_config.get('file_type', 'excel')
    smiles_column = data_config.get('smiles_column', 'SMILES')
    target_column = data_config.get('target_column', 'Value')
    global_features_column = data_config.get('global_features_column', 'T')
    split_column = data_config.get('split_column', 'Subset')
    split_mapping = data_config.get('split_mapping', {'Training': 0, 'Validation': 1, 'Test': 2})

    # Read the file
    if file_type == 'csv':
        df = pd.read_csv(file_name)
    elif file_type == 'excel':
        df = pd.read_excel(file_name)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    print(f"Loading data from: {file_name}")

    smiles = df[smiles_column].to_numpy()
    targets = df[target_column].to_numpy()

    tags = df[split_column].to_numpy()
    custom_split = np.array([split_mapping[tag] for tag in tags])

    global_feats = df[global_features_column].to_numpy()

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
        fragmentation=fragmentation,
    )

    # Split data
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data, mean_target, std_target

def trainable(config, device, fragmentation, data_config):
    # Load the ICP dataset
    train_data, val_data, _, mean_target, std_target = load_icp_dataset(fragmentation, data_config)

    # Load the model
    frag_dim = fragmentation.frag_dim

    mlp_layers = int(config['MLP_layers'])
    mlp_hidden = return_hidden_layers(mlp_layers)
    net_params = {
        "device": device,
        "num_atom_type": 44,
        "num_bond_type": 12,
        "dropout": config['dropout'],
        "MLP_layers": mlp_layers,
        "frag_dim": frag_dim,
        "final_dropout": config['final_dropout'],
        "use_global_features": True,
        "num_heads": 1,
        "node_in_dim": 44,
        "edge_in_dim": 12,
        "num_global_feats": 1,
        "hidden_dim": int(config['hidden_dim']),
        "mlp_out_hidden": mlp_hidden,
        "num_layers_atom": int(config['num_layers_atom']),
        "num_layers_mol": int(config['num_layers_mol']),
        # L1 parameters
        "L1_layers_atom": int(config['L1_layers_atom']),
        "L1_layers_mol": int(config['L1_layers_mol']),
        "L1_dropout": config['L1_dropout'],
        "L1_hidden_dim": int(config['L1_hidden_dim']),
        # L2 parameters
        "L2_layers_atom": int(config['L2_layers_atom']),
        "L2_layers_mol": int(config['L2_layers_mol']),
        "L2_dropout": config['L2_dropout'],
        "L2_hidden_dim": int(config['L2_hidden_dim']),
        # L3 parameters
        "L3_layers_atom": int(config['L3_layers_atom']),
        "L3_layers_mol": int(config['L3_layers_mol']),
        "L3_dropout": config['L3_dropout'],
        "L3_hidden_dim": int(config['L3_hidden_dim']),
        "output_dim": 1,
    }

    model = GroupGAT_ICP(net_params)
    model.to(device)

    # Set up optimizer, scheduler, and early stopper
    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    early_Stopper = EarlyStopping(patience=30, model_name='GroupGAT_ICP', skip_save=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'], min_lr=1e-10, patience=10)
    loss_function = torch.nn.functional.mse_loss

    # Create data loaders
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    iterations = 1000
    start_epoch = 0

    # Handle checkpointing
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state_dict = torch.load(
                os.path.join(checkpoint_dir, "model.pt"),
                map_location=device,
            )
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
            start_epoch = torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))["epoch"] + 1

    model.train()
    for i in range(start_epoch, iterations):
        train_loss = train_epoch_jittable(model=model, loss_func=loss_function, optimizer=optimizer, train_loader=train_loader, device=device)
        val_loss = val_epoch_jittable(model=model, loss_func=loss_function, val_loader=val_loader, device=device)
        scheduler.step(val_loss)
        early_Stopper(val_loss=val_loss, model=model)

        metrics = {
            "epoch": i + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mse_loss": val_loss
        }
        train.report(metrics)

        should_checkpoint = i % config.get("checkpoint_freq", 15) == 0
        if should_checkpoint and train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, "optimizer.pt"))
                torch.save({"epoch": i}, os.path.join(temp_checkpoint_dir, "extra_state.pt"))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(metrics, checkpoint=checkpoint)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=100,
                        help='the number of samples/instances that will be running (default: %(default)s)')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the config file')

    args = parser.parse_args()
    n_samples = args.samples
    config_file = args.config_file  # Get the config file from arguments

    # Load and render the YAML template with Jinja2
    with open(config_file, 'r') as f:
        template = Template(f.read())
        # First, parse the unrendered config to get the variables
        f.seek(0)
        f_config_unparsed = yaml.safe_load(f)
        template_vars = {
            'root_path': f_config_unparsed.get('root_path', '')  # Default to empty string if not defined
        }
        rendered_yaml = template.render(template_vars)
        data_config = yaml.safe_load(rendered_yaml)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = 1 if torch.cuda.is_available() else 0

    # Initialize fragmentation
    fragmentation_scheme = data_config.get('fragmentation_scheme', 'MG_plus_reference')
    fragmentation_save_path = data_config.get('fragmentation_save_file_path', 'ICP_fragmentation.pth')
    fragmentation_verbose = data_config.get('fragmentation_verbose', False)

    print("Initializing fragmentation...")
    fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path=fragmentation_save_path, verbose=fragmentation_verbose)
    print("Done.")

    # Define the hyperparameter search space
    config_space = CS.ConfigurationSpace()
    config_space.add([
        CS.UniformFloatHyperparameter('initial_lr', lower=1e-5, upper=1e-1, log=True),
        CS.UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-1, log=True),
        CS.UniformFloatHyperparameter("lr_reduction_factor", lower=0.4, upper=0.99),
        CS.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.5),
        CS.UniformFloatHyperparameter("final_dropout", lower=0.0, upper=0.5),
        CS.UniformIntegerHyperparameter("hidden_dim", lower=64, upper=512),
        CS.UniformIntegerHyperparameter("num_layers_atom", lower=1, upper=5),
        CS.UniformIntegerHyperparameter("num_layers_mol", lower=1, upper=5),
        CS.UniformIntegerHyperparameter("MLP_layers", lower=1, upper=3),
        # L1 parameters
        CS.UniformIntegerHyperparameter("L1_layers_atom", lower=1, upper=5),
        CS.UniformIntegerHyperparameter("L1_layers_mol", lower=1, upper=5),
        CS.UniformFloatHyperparameter("L1_dropout", lower=0.0, upper=0.5),
        CS.UniformIntegerHyperparameter("L1_hidden_dim", lower=32, upper=256),
        # L2 parameters
        CS.UniformIntegerHyperparameter("L2_layers_atom", lower=1, upper=5),
        CS.UniformIntegerHyperparameter("L2_layers_mol", lower=1, upper=5),
        CS.UniformFloatHyperparameter("L2_dropout", lower=0.0, upper=0.5),
        CS.UniformIntegerHyperparameter("L2_hidden_dim", lower=32, upper=256),
        # L3 parameters
        CS.UniformIntegerHyperparameter("L3_layers_atom", lower=1, upper=5),
        CS.UniformIntegerHyperparameter("L3_layers_mol", lower=1, upper=5),
        CS.UniformFloatHyperparameter("L3_dropout", lower=0.0, upper=0.5),
        CS.UniformIntegerHyperparameter("L3_hidden_dim", lower=32, upper=256),
    ])

    # Prepare the trainable function
    my_trainable = partial(trainable, device=device, fragmentation=fragmentation, data_config=data_config)
    trainable_with_resources = tune.with_resources(my_trainable, {"cpu": 6, "gpu": gpu})

    # Define search algorithm
    algo = TuneBOHB(config_space, metric="mse_loss", mode='min')

    # Get the trial control algorithm
    scheduler = HyperBandForBOHB(
        time_attr="epoch",
        max_t=1000,
        reduction_factor=2.78,
    )

    # Define the metrics and parameters to report
    from ray.tune import CLIReporter
    reporter = CLIReporter(
        parameter_columns=["initial_lr", "weight_decay", "hidden_dim"],
        metric_columns=["epoch", "val_loss", "mse_loss"]
    )

    # Initialize the tuner
    tuner = Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=algo,
            num_samples=n_samples,
            metric="mse_loss",
            mode="min",
        ),
        run_config=RunConfig(
            name="groupgat_icp_hpo",
            stop={"epoch": 1000},
            progress_reporter=reporter
        ),
    )

    result = tuner.fit()

    # Save the best results
    best_result = result.get_best_result(metric="mse_loss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics

    results_to_save = {
        "best_config": best_config,
        "best_metrics": best_metrics
    }

    directory = os.path.join("env", "bohb_results")
    file_name = os.path.join(directory, f"best_hyperparameters_GroupGAT_ICP.json")
    os.makedirs(directory, exist_ok=True)
    with open(file_name, "w") as file:
        import json
        json.dump(results_to_save, file, indent=4)