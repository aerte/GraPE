import torch
from grape_chem.datasets import FreeSolv
from grape_chem.utils.model_utils import set_seed
from grape_chem.utils import RevIndexedSubSet
from grape_chem.utils.ensemble import Bagging, RandomWeightInitialization, Jackknife, BayesianBootstrap
from grape_chem.logging.mlflow_logging import setup_mlflow
from grape_chem.utils.data import load_data_from_csv, get_path, get_model_dir
from grape_chem.logging.config import load_config
import time
import numpy as np
import mlflow
import mlflow.pytorch
import yaml
import os
from typing import Dict

# Absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def print_averages(metrics, technique):
    avg_metrics = {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}
    print(f"""Average Metrics for Ensemble Models using {technique}:
    MSE: {avg_metrics['mse']:.6f}
    RMSE: {avg_metrics['rmse']:.6f}
    SSE: {avg_metrics['sse']:.6f}
    MAE: {avg_metrics['mae']:.6f}
    R2: {avg_metrics['r2']:.6f}
    MRE: {avg_metrics['mre']:.6f}
    MDAPE: {avg_metrics['mdape']:.6f}
    """)

def calculate_std_metrics(metrics_dict, technique):
    metrics = np.array([
        [metric['mse'], metric['rmse'], metric['sse'], metric['mae'], 
         metric['r2'], metric['mre'], metric['mdape']]
        for metric in metrics_dict
    ])
    std_metrics = {
        f'{technique} mse std': np.std(metrics[:, 0]),
        f'{technique} rmse std': np.std(metrics[:, 1]),
        f'{technique} sse std': np.std(metrics[:, 2]),
        f'{technique} mae std': np.std(metrics[:, 3]),
        f'{technique} r2 std': np.std(metrics[:, 4]),
        f'{technique} mre std': np.std(metrics[:, 5]),
        f'{technique} mdape std': np.std(metrics[:, 6])
    }
    return std_metrics

def log_average_ensemble(metrics, technique):
    avg_metrics = {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}
    avg_metrics = {f'{technique} {k}': v for k, v in avg_metrics.items()}
    mlflow.log_metrics(avg_metrics)

def run_ensemble(config: Dict):
    with setup_mlflow(config):
        for seed in config['seed']:
            ### SETUP ###
            print(f"\nSeed {seed}:")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_data, val_data, test_data, data = load_data_from_csv(config, return_dataset=True)
            node_in_dim = data[0].x.shape[1]
            edge_in_dim = data[0].edge_attr.shape[1]
            #train_data, val_data, test_data, node_in_dim, edge_in_dim, device = setup_data(config, seed=seed)
            
            ### TRAINING ###
            start_time = time.time()
            bagging_metrics = Bagging(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, config).run()
            bagging_time = time.time() - start_time

            start_time = time.time()
            random_metrics = RandomWeightInitialization(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, config).run()
            random_weights_time = time.time() - start_time

            start_time = time.time()
            jackknife_metrics = Jackknife(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, config).run()
            jackknife_time = time.time() - start_time

            start_time = time.time()
            bb_metrics = BayesianBootstrap(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, config).run()
            bb_time = time.time() - start_time


            print("#####################################################################")
            print("Completed with model: ", config['model'].upper())

            if config['Print Metrics']:
                print("Bagging Metrics:", bagging_metrics)
                print("#####################################################################")
                print_averages(bagging_metrics, 'Bagging')
                
                print("\nBagging Metrics Std Dev:")
                #print(calculate_std_metrics(bagging_metrics, 'Bagging'), "\n")
                print("Bagging Time: ", bagging_time, "\n")
                print_averages(random_metrics, 'Random Initialization')

                print("\nRandom Initialization Metrics Std Dev:")
                #print(calculate_std_metrics(random_metrics), "\n")
                print("Random Initialization Time: ", random_weights_time, "\n")
                print_averages(jackknife_metrics, 'Jackknife')
                
                print("\nJackknife Metrics Std Dev:")
                #print(calculate_std_metrics(jackknife_metrics), "\n")
                print("Jackknife Time: ", jackknife_time, "\n")
                print_averages(bb_metrics, 'Weighted Bayesian Bootstrap')

                print("\nWeighted Bayesian Bootstrap Metrics Std Dev:")
                #print(calculate_std_metrics(bb_metrics), "\n")
                print("Weighted Bayesian Bootstrap Time: ", bb_time, "\n")
                print("#####################################################################")
            

            mlflow.log_metrics(calculate_std_metrics(bagging_metrics, 'Bagging'))
            log_average_ensemble(bagging_metrics, 'Bagging')
            
            mlflow.log_metrics(calculate_std_metrics(random_metrics, 'Random Initialization'))
            log_average_ensemble(random_metrics, 'Random Initialization')

            mlflow.log_metrics(calculate_std_metrics(jackknife_metrics, 'Jackknife'))
            log_average_ensemble(jackknife_metrics, 'Jackknife')

            mlflow.log_metrics(calculate_std_metrics(bb_metrics, 'Weighted Bayesian Bootstrap'))
            log_average_ensemble(bb_metrics, 'Weighted Bayesian Bootstrap')

def main():
    # Load config file from its path, in this case its in the same directory as the script
    path = get_path(current_dir, 'config.yaml')
    base_config = load_config(path)
    
    # Define configurations for the search space
    #time = datetime.datetime.now().strftime("%d-%m-%Y-%H")
    search_space = {
        'epochs': 300,
        'batch_size': None,  # Set in setup_data
        'n_models': 10,
        'early_stopping_patience': 50,
        'scheduler_patience': 5,
        'scheduler_factor': 0.9,
        'learning_rate': 1e-3,
        'seed': [42],
        'weight_seed': 100,
        'Print Metrics': False,
        'model': 'afp',  # 'afp' or 'dmpnn'
        'model_name': 'afp',
        'run_name': 'ensemble',
        'model_seed': 42,
    }

    config = {**base_config, **search_space}

    run_ensemble(config)

if __name__ == "__main__":
    main()