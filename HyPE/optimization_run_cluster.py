import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from functools import partial
from grape_chem.utils import DataSet, split_data, train_epoch_jittable, val_epoch_jittable
from grape_chem.utils import JT_SubGraph
from grape_chem.models import AFP, MPNN, DMPNN, MEGNet, GroupGAT_jittable
from grape_chem.utils import EarlyStopping
import ConfigSpace as CS
from ray import tune, train
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.air import RunConfig
from ray.train import Checkpoint
from ray.tune import Tuner
from ray.tune import CLIReporter

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

def return_hidden_layers(num):
    """ Returns a list of hidden layers, starting from 2**num*32, reducing the hidden dim by 2 every step. """
    return [2 ** i * 32 for i in range(num, 0, -1)]

def load_dataset_from_file(data_config):
    """
    Loads the dataset based on the configuration provided.

    Parameters:
    -----------
    data_config: dict
        Dictionary containing the data configuration parameters.
        Expected keys:
        - dataset_name: str, name of the dataset (for logging purposes)
        - file_name: str, path to the data file
        - file_type: str, 'csv' or 'excel'
        - smiles_column: str, name of the SMILES column
        - target_column: str or list of str, name(s) of the target column(s)
        - global_features_column: str or list of str, name(s) of the global features column(s) (optional)
        - split_column: str, name of the column containing split information (optional)
        - split_ratios: list of float, ratios for train/val/test split if split_column is not provided (optional)
        - fragmentation: dict, fragmentation settings (optional)
    """
    # Read the file
    file_name = data_config.get('file_name')
    file_type = data_config.get('file_type', 'csv')

    if file_type == 'csv':
        df = pd.read_csv(file_name)
    elif file_type == 'excel':
        sheet_name = data_config.get('sheet_name', 0)
        df = pd.read_excel(file_name, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Extract SMILES, target, global features
    smiles_column = data_config.get('smiles_column', 'SMILES')
    target_column = data_config.get('target_column', 'Target')
    global_features_column = data_config.get('global_features_column', None)
    split_column = data_config.get('split_column', None)

    smiles = df[smiles_column].to_numpy()
    target = df[target_column].to_numpy()

    if global_features_column is not None:
        global_feats = df[global_features_column].to_numpy()
    else:
        # If no global features are provided, generate random ones
        global_feats = np.random.randn(len(smiles))

    # Standardize targets and global features
    mean_target, std_target = np.mean(target), np.std(target)
    target = (target - mean_target) / std_target
    mean_global_feats, std_global_feats = np.mean(global_feats), np.std(global_feats)
    global_feats = (global_feats - mean_global_feats) / std_global_feats

    # Handle fragmentation
    fragmentation_settings = data_config.get('fragmentation', None)
    if fragmentation_settings is not None:
        # Get fragmentation settings
        scheme = fragmentation_settings.get('scheme', None)
        save_file_path = fragmentation_settings.get('save_file_path', None)
        verbose = fragmentation_settings.get('verbose', False)
        fragmentation = JT_SubGraph(scheme=scheme, save_file_path=save_file_path, verbose=verbose)
    else:
        fragmentation = None

    # Handle splitting
    if split_column is not None:
        split_values = df[split_column].to_numpy()
        # Map split values to integers
        unique_splits = np.unique(split_values)
        split_mapping = {split: idx for idx, split in enumerate(unique_splits)}
        custom_split = np.array([split_mapping[split] for split in split_values])

    else:
        # If no split column provided, split using ratio
        split_ratios = data_config.get('split_ratios', [0.8, 0.1, 0.1])
        assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(smiles))
        train_idx, test_idx = train_test_split(indices, test_size=split_ratios[2], random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]), random_state=42)
        custom_split = np.zeros(len(smiles), dtype=int)
        custom_split[val_idx] = 1
        custom_split[test_idx] = 2

    # Hacky way to filter things out
    # the filter function is not working properly,
    # so we use a dataset without fragmentation to filter out the data.

    # Initialize dataset
    data = DataSet(
        smiles=smiles,
        target=target,
        global_features=global_feats,
        filter=True,
        fragmentation=fragmentation,
        custom_split=custom_split
    )
    print("len(smiles):", len(smiles))
    print("len(target):", len(target))
    print("len(global_feats):", len(global_feats))
    print("len(custom_split):", len(custom_split))
    # Split data
    train_set, val_set, test_set = split_data(
        data, split_type='custom', custom_split=data.custom_split
    )

    return train_set, val_set, test_set, mean_target, std_target

def load_model(model_name, config, device=None):
    """ Function to load a model based on a model name and a config dictionary. """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mlp_out = return_hidden_layers(int(config['MLP_layers']))

    if model_name == "AFP":
        return AFP(node_in_dim=44, edge_in_dim=12, num_layers_mol=int(config["afp_mol_layers"]),
                   num_layers_atom=int(config["depth"]), rep_dropout=config["dropout"],
                   hidden_dim=int(config["gnn_hidden_dim"]),
                   mlp_out_hidden=mlp_out)
    elif model_name == "MPNN":
        return MPNN(node_in_dim=44, edge_in_dim=12, num_layers=int(config["depth"]),
                    mlp_out_hidden=mlp_out, rep_dropout=config["dropout"],
                    node_hidden_dim=int(config["gnn_hidden_dim"]))
    elif model_name == "DMPNN":
        return DMPNN(node_in_dim=44, edge_in_dim=12, node_hidden_dim=int(config["gnn_hidden_dim"]),
                     depth=int(config["depth"]), dropout=0, mlp_out_hidden=mlp_out,
                     rep_dropout=config["dropout"])
    elif model_name == "MEGNet":
        return MEGNet(node_in_dim=44, edge_in_dim=12, global_in_dim=1, node_hidden_dim=int(config["gnn_hidden_dim"]),
                      edge_hidden_dim=int(config["edge_hidden_dim"]), depth=int(config["depth"]),
                      mlp_out_hidden=mlp_out, rep_dropout=config["dropout"],
                      device=device)
    elif model_name == "GroupGAT":
        net_params = {
            "device": device,
            "num_atom_type": 44,
            "num_bond_type": 12,
            "dropout": config["dropout"],
            "MLP_layers": int(config["MLP_layers"]),
            "frag_dim": 219,  # Update this if necessary
            "final_dropout": config.get("final_dropout", 0.257507),
            "num_heads": 1,
            "node_in_dim": 44,
            "edge_in_dim": 12,
            "num_global_feats": 1,
            "hidden_dim": int(config["hidden_dim"]),
            "mlp_out_hidden": mlp_out,
            "num_layers_atom": int(config["num_layers_atom"]),
            "num_layers_mol": int(config["num_layers_mol"]),
            "L1_layers_atom": int(config["L1_layers_atom"]),
            "L1_layers_mol": int(config["L1_layers_mol"]),
            "L1_dropout": config["L1_dropout"],
            "L1_hidden_dim": int(config["L1_hidden_dim"]),

            "L2_layers_atom": int(config["L2_layers_atom"]),
            "L2_layers_mol": int(config["L2_layers_mol"]),
            "L2_dropout": config["L2_dropout"],
            "L2_hidden_dim": int(config["L2_hidden_dim"]),

            "L3_layers_atom": int(config["L3_layers_atom"]),
            "L3_layers_mol": int(config["L3_layers_mol"]),
            "L3_dropout": config["L3_dropout"],
            "L3_hidden_dim": int(config["L3_hidden_dim"]),
        }
        return GroupGAT_jittable.GCGAT_v4pro_jit(net_params)

def trainable(config, data_config=None, model_name: str = None, is_dmpnn: bool = False,
              device: torch.device = None, is_megnet: bool = False, is_groupgat: bool = True,):
    """
    The trainable for Ray Tune.

    Parameters
    -----------
        config: dict
            A ConfigSpace dictionary adhering to the required parameters in the trainable.
        data_config: dict
            Dictionary containing the data configuration parameters.
        model_name: str
            The model to be loaded.
    """
    # Load the dataset
    train_set, val_set, _, _, _ = load_dataset_from_file(data_config)

    # Define the model
    model = load_model(model_name=model_name, config=config, device=device)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    early_Stopper = EarlyStopping(patience=30, model_name='random', skip_save=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'],
                                               min_lr=1e-10, patience=10)
    loss_function = torch.nn.functional.mse_loss

    train_data = DataLoader(train_set, batch_size=300)
    val_data = DataLoader(val_set, batch_size=300)

    iterations = 400
    start_epoch = 0
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
        train_loss = train_epoch_jittable(model=model, loss_func=loss_function, optimizer=optimizer, train_loader=train_data,
                                 device=device)
        val_loss = val_epoch_jittable(model=model, loss_func=loss_function, val_loader=val_data, device=device)
        scheduler.step(val_loss)
        early_Stopper(val_loss=val_loss, model=model)

        # Report metrics to Ray Tune
        metrics = {
            "epoch": i + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mse_loss": val_loss
        }
        train.report(metrics)

        # Checkpointing
        should_checkpoint = i % config.get("checkpoint_freq", 15) == 0
        if should_checkpoint and train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                # Save model and optimizer state
                torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, "optimizer.pt"))
                torch.save({"epoch": i}, os.path.join(temp_checkpoint_dir, "extra_state.pt"))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Report checkpoint to Ray Tune
            train.report(metrics, checkpoint=checkpoint)

if __name__ == '__main__':
    import argparse
    import yaml
    import tempfile

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--samples', type=int, default=100,
                        help='the number of samples/instances that will be running (default: %(default)s)')
    # Add the new resume flag
    parser.add_argument('--resume_from_checkpoint_if_present', action='store_true',
                        help='Resume the experiment from checkpoint if this flag is set')
    args = parser.parse_args()
    n_samples = args.samples

    # Read the configuration file
    with open(args.config_file, 'r') as f:
        data_config = yaml.safe_load(f)

    # Extract the resume path from the config file if it exists
    resume_from_path = data_config.get('resume_from_path', None)

    # Set up model and device configurations
    model_name = "GroupGAT"
    is_dmpnn = False
    is_megnet = False
    is_groupgat = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = 1
    else:
        device = torch.device("cpu")
        gpu = 0

# Define the configuration space for hyperparameter optimisation
    config_space = CS.ConfigurationSpace()
    config_space.add(CS.UniformIntegerHyperparameter("depth", lower=1, upper=4))
    config_space.add(CS.UniformFloatHyperparameter('initial_lr', lower=1e-4, upper=1e-1))
    config_space.add(CS.UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-2))
    config_space.add(CS.UniformFloatHyperparameter("lr_reduction_factor", lower=0.4, upper=0.95))
    config_space.add(CS.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.15))
    config_space.add(CS.UniformIntegerHyperparameter("MLP_layers", lower=1, upper=3))
    # Additional parameters for GroupGAT
    config_space.add(CS.UniformIntegerHyperparameter("hidden_dim", lower=64, upper=256))
    config_space.add(CS.UniformIntegerHyperparameter("num_layers_atom", lower=1, upper=4))
    config_space.add(CS.UniformIntegerHyperparameter("num_layers_mol", lower=1, upper=4))
    config_space.add(CS.UniformFloatHyperparameter("final_dropout", lower=0.0, upper=0.15))
    config_space.add(CS.UniformIntegerHyperparameter("num_heads", lower=1, upper=4))
    # L1:
    config_space.add(CS.UniformIntegerHyperparameter("L1_layers_atom", lower=1, upper=4))
    config_space.add(CS.UniformIntegerHyperparameter("L1_layers_mol", lower=1, upper=4))
    config_space.add(CS.UniformFloatHyperparameter("L1_dropout", lower=0.0, upper=0.15))
    config_space.add(CS.UniformIntegerHyperparameter("L1_hidden_dim", lower=32, upper=256))
    config_space.add(CS.UniformIntegerHyperparameter("L2_layers_atom", lower=1, upper=4))
    config_space.add(CS.UniformIntegerHyperparameter("L2_layers_mol", lower=1, upper=4))
    config_space.add(CS.UniformFloatHyperparameter("L2_dropout", lower=0.0, upper=0.15))
    config_space.add(CS.UniformIntegerHyperparameter("L2_hidden_dim", lower=32, upper=256))
    config_space.add(CS.UniformIntegerHyperparameter("L3_layers_atom", lower=1, upper=4))
    config_space.add(CS.UniformIntegerHyperparameter("L3_layers_mol", lower=1, upper=4))
    config_space.add(CS.UniformFloatHyperparameter("L3_dropout", lower=0.0, upper=0.15))
    config_space.add(CS.UniformIntegerHyperparameter("L3_hidden_dim", lower=32, upper=256))

    # Prepare the trainable function with partial
    my_trainable = partial(trainable, data_config=data_config, model_name=model_name, is_dmpnn=is_dmpnn,
                           is_megnet=is_megnet, device=device, is_groupgat=is_groupgat)

    trainable_with_resources = tune.with_resources(my_trainable, {"cpu": 6, "gpu": gpu})

    # Define search algorithm and scheduler
    algo = TuneBOHB(config_space, mode='min', metric="mse_loss")
    scheduler = HyperBandForBOHB(
        time_attr="epoch",
        max_t=1000,
        reduction_factor=2.78,
    )

    # Define the metrics and parameters to report
    reporter = CLIReporter(
        parameter_columns=["depth", "hidden_dim", "initial_lr", "dropout"],
        metric_columns=["epoch", "train_loss", "val_loss", "mse_loss"]
    )

    storage_path = resume_from_path if resume_from_path else None

    if args.resume_from_checkpoint_if_present and resume_from_path:
        # Restore the tuner from the previous experiment
        tuner = Tuner.restore(
            path=resume_from_path,
            trainable=trainable_with_resources,
            resume_errored=True,  # Optional: resumes any errored trials
        )
    else:
        # Initialize a new tuner
        tuner = Tuner(
            trainable_with_resources,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                search_alg=algo,
                mode='min',
                metric="mse_loss",
                num_samples=n_samples
            ),
            run_config=RunConfig(
                name="bo_exp",
                stop={"epoch": 600},
                progress_reporter=reporter,
                storage_path=storage_path,  # Use storage_path instead of local_dir
            ),
        )

    result = tuner.fit()

    import json

    best_result = result.get_best_result(metric="mse_loss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics

    results_to_save = {
        "best_config": best_config,
        "best_metrics": best_metrics
    }

    # Save the results
    directory = os.environ.get('BOHB_RESULTS_DIR', os.path.join("env", "bohb_results"))
    dataset_name = data_config.get('dataset_name', 'dataset')
    file_name = os.path.join(directory, f"new_best_hyperparameters_{model_name}_{dataset_name}.json")
    os.makedirs(directory, exist_ok=True)
    with open(file_name, "w") as file:
        json.dump(results_to_save, file, indent=4)
