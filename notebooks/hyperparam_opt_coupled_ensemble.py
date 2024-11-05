import torch
import torch.nn as nn
from grape_chem.models import GroupGAT_Ensemble
from grape_chem.utils import (
    DataSet, EarlyStopping, train_epoch, val_epoch,
    set_seed, JT_SubGraph
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

# Initialize Ray
ray.init()

# Set seed for reproducibility
set_seed(42)

def standardize(x, mean, std):
    return (x - mean) / std

def split_data_by_indices(data, train_idx, val_idx, test_idx):
    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]
    return train, val, test

def load_icp_dataset(fragmentation):
    import os
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

    return train_data, val_data, test_data, mean_target, std_target

def return_hidden_layers(num):
    """ Returns a list of hidden layers, starting from 2**num*32, reducing the hidden dim by 2 every step.

    Example
    --------

    >>>return_hidden_layers(3)

    [256, 128, 64]
    """
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

def trainable(config, device, fragmentation):
    # Load the ICP dataset
    train_data, val_data, _, mean_target, std_target = load_icp_dataset(fragmentation)

    # Load the model
    frag_dim = fragmentation.frag_dim
    model = load_model(config, device, frag_dim)
    model.to(device)

    # Set up optimizer, scheduler, and early stopper
    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    early_Stopper = EarlyStopping(patience=30, model_name='random', skip_save=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'], min_lr=1e-10, patience=10)
    loss_function = torch.nn.functional.mse_loss

    # Create data loaders
    batch_size = config.get('batch_size', 32)  # Adjust batch size if necessary
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    iterations = 1000  # Adjust as necessary
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
        train_loss = train_epoch(model=model, loss_func=loss_function, optimizer=optimizer, train_loader=train_loader, device=device)
        val_loss = val_epoch(model=model, loss_func=loss_function, val_loader=val_loader, device=device)
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

    args = parser.parse_args()
    n_samples = args.samples

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = 1 if torch.cuda.is_available() else 0

    # Initialize fragmentation
    fragmentation_scheme = "MG_plus_reference"
    print("Initializing fragmentation...")
    fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path='/home/paul/Documents/spaghetti/GraPE/env/ICP_fragmentation.pth')
    print("Done.")

    # Define the hyperparameter search space
    config_space = CS.ConfigurationSpace()
    config_space.add([
        CS.UniformFloatHyperparameter('initial_lr', lower=1e-5, upper=1e-1),
        CS.UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-1),
        CS.UniformFloatHyperparameter("lr_reduction_factor", lower=0.4, upper=0.99),
        CS.UniformFloatHyperparameter("final_dropout", lower=0.0, upper=0.5),
    ])

    model_names = ['A', 'B', 'C', 'D', 'E']
    for model in model_names:
        config_space.add([
            CS.UniformIntegerHyperparameter(f"{model}_hidden_dim", lower=64, upper=512),
            CS.UniformIntegerHyperparameter(f"{model}_num_layers_atom", lower=1, upper=5),
            CS.UniformIntegerHyperparameter(f"{model}_num_layers_mol", lower=1, upper=5),
            CS.UniformFloatHyperparameter(f"{model}_dropout", lower=0.0, upper=0.15),
            CS.UniformIntegerHyperparameter(f"{model}_MLP_layers", lower=1, upper=3),
            CS.UniformFloatHyperparameter(f"{model}_final_dropout", lower=0.0, upper=0.5),
            # L1 parameters
            CS.UniformIntegerHyperparameter(f"{model}_L1_layers_atom", lower=1, upper=5),
            CS.UniformIntegerHyperparameter(f"{model}_L1_layers_mol", lower=1, upper=5),
            CS.UniformFloatHyperparameter(f"{model}_L1_dropout", lower=0.0, upper=0.5),
            CS.UniformIntegerHyperparameter(f"{model}_L1_hidden_dim", lower=32, upper=256),
            # L2 parameters
            CS.UniformIntegerHyperparameter(f"{model}_L2_layers_atom", lower=1, upper=5),
            CS.UniformIntegerHyperparameter(f"{model}_L2_layers_mol", lower=1, upper=5),
            CS.UniformFloatHyperparameter(f"{model}_L2_dropout", lower=0.0, upper=0.5),
            CS.UniformIntegerHyperparameter(f"{model}_L2_hidden_dim", lower=32, upper=256),
            # L3 parameters
            CS.UniformIntegerHyperparameter(f"{model}_L3_layers_atom", lower=1, upper=5),
            CS.UniformIntegerHyperparameter(f"{model}_L3_layers_mol", lower=1, upper=5),
            CS.UniformFloatHyperparameter(f"{model}_L3_dropout", lower=0.0, upper=0.5),
            CS.UniformIntegerHyperparameter(f"{model}_L3_hidden_dim", lower=32, upper=256),
        ])

    # Prepare the trainable function
    my_trainable = partial(trainable, device=device, fragmentation=fragmentation)
    trainable_with_resources = tune.with_resources(my_trainable, {"cpu": 6, "gpu": gpu})

    # Define search algorithm
    algo = TuneBOHB(config_space, metric="mse_loss", mode='min')

    # Get the trial control algorithm
    scheduler = HyperBandForBOHB(
        time_attr="epoch",
        max_t=1000,  # Adjust as necessary
        reduction_factor=2.78,
    )

    # Define the metrics and parameters to report
    from ray.tune import CLIReporter
    reporter = CLIReporter(
        parameter_columns=["initial_lr", "weight_decay"],
        metric_columns=["epoch", "train_loss", "val_loss", "mse_loss"]
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
            name="groupgat_ensemble_icp_hpo",
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
    file_name = os.path.join(directory, f"best_hyperparameters_GroupGAT_Ensemble_ICP.json")
    os.makedirs(directory, exist_ok=True)
    with open(file_name, "w") as file:
        import json
        json.dump(results_to_save, file, indent=4)