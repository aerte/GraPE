import torch
import time
import numpy as np
import mlflow
import mlflow.pytorch
import yaml
import os
from typing import Dict
import matplotlib.pyplot as plt

from grape_chem.utils.model_utils import set_seed
from grape_chem.utils.ensemble import Bagging, RandomWeightInitialization, Jackknife, BayesianBootstrap
from grape_chem.utils.data import load_dataset_from_csv, get_path
from grape_chem.logging.config import load_config
from grape_chem.logging.mlflow_logging import setup_mlflow, track_params

# Absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def run_ensemble(config: Dict, data_bundle: Dict):
    with setup_mlflow(config):
        track_params(config, data_bundle)
        ### SETUP ###
        set_seed(config["seed"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df, train_data, val_data, test_data, data = data_bundle['df'], data_bundle['train_data'], data_bundle['val_data'], data_bundle['test_data'], data_bundle['data']
        node_in_dim = data[0].x.shape[1]
        edge_in_dim = data[0].edge_attr.shape[1]

        ### ENSEMBLE TECHNIQUES ###
        techniques = {
            "Bagging": Bagging,
            "Random Initialization": RandomWeightInitialization,
            #"Jackknife": Jackknife,
            "Weighted Bayesian Bootstrap": BayesianBootstrap
        }

        for technique_name, EnsembleClass in techniques.items():
            start_time = time.time()
            # Important to add the Dataset data to the run function in order to scale the data
            EnsembleClass(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, config).run(data)
            elapsed_time = time.time() - start_time

            mlflow.log_param(f"Elapsed Time {technique_name}", elapsed_time)

def main():
    # Load config file from its path, in this case its in the same directory as the script
    path = get_path(current_dir, 'config.yaml')
    base_config = load_config(path)
    
    # Define configurations for the search space
    search_space = {
        'epochs': 10,
        'batch_size': None,  # Set in setup_data tune.choice([1024,4096,8192])
        'n_models': 4,
        'early_stopping_patience': 25,
        'scheduler_patience': 10,
        'scheduler_factor': 0.8,
        'learning_rate': 0.0014486833887239338,
        'weight_decay': 1.0005268542378315e-05,
        'dropout': 0.03745401188473625,
        'seed': 42,
        'weight_seed': 100,
        'hidden_dim': 128,
        'mlp_layers': 2,
        'num_layers_atom': 2,
        'num_layers_mol': 2,
        'Print Metrics': False,
        'model': 'afp',  # 'afp' or 'dmpnn' REQUIRED FOR ensemble.py def create_model(self):
        'model_name': 'afp',
        'run_name': 'ensemble',
        'model_seed': 42,
    }

    base_config['save_path'] = get_path(current_dir, 'ensemble')

    config = {**base_config, **search_space}

    df, train_data, val_data, test_data, data = load_dataset_from_csv(config, return_dataset=True, limit = None) # None 100
    # Bundle the df & datasets into one variable
    data_bundle = {
        'df': df,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'data': data,
    }

    run_ensemble(config, data_bundle)

if __name__ == "__main__":
    main()
