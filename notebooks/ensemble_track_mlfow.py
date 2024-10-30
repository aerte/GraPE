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
from grape_chem.logging.mlflow_logging import setup_mlflow

# Absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def run_ensemble(config: Dict):
    with setup_mlflow(config):
        ### SETUP ###
        set_seed(config["seed"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df, train_data, val_data, test_data, data = load_dataset_from_csv(config, return_dataset=True)
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
        'epochs': 100,
        'batch_size': None,  # Set in setup_data
        'n_models': 5,
        'early_stopping_patience': 25,
        'scheduler_patience': 5,
        'scheduler_factor': 0.9,
        'learning_rate': 1e-3,
        'seed': 42,
        'weight_seed': 100,
        'Print Metrics': False,
        'model': 'afp',  # 'afp' or 'dmpnn' REQUIRED FOR ensemble.py def create_model(self):
        'model_name': 'afp',
        'run_name': 'ensemble',
        'model_seed': 42,
    }

    base_config['save_path'] = get_path(current_dir, 'ensemble')

    config = {**base_config, **search_space}
    run_ensemble(config)

if __name__ == "__main__":
    main()
