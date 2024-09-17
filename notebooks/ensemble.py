import torch
from grape_chem.datasets import FreeSolv
from grape_chem.utils.model_utils import set_seed
from grape_chem.utils import RevIndexedSubSet
from grape_chem.utils.ensemble import Bagging, RandomWeightInitialization, Jackknife, BayesianBootstrap
import time
import numpy as np

# Hyperparameters
HYPERPARAMS = {
    'learning_rate': 0.002,
    'batch_size': None,  # Set in setup_data
    'epochs': 1,
    'n_models': 10,
    'early_stopping_patience': 50,
    'scheduler_patience': 5,
    'scheduler_factor': 0.9,
    'seed': [42],
    'weight_seed': 100,
    'Print Metrics': True,
    'model': 'dmpnn',  # 'afp' or 'dmpnn'
    'chemprop': True
}

# Chemprop allowed atoms and features
allowed_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'I']
atom_feature_list = ['chemprop_atom_features']
bond_feature_list = ['chemprop_bond_features']

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

def calculate_std_metrics(metrics_dict):
    metrics = np.array([
        [metric['mse'], metric['rmse'], metric['sse'], metric['mae'], 
         metric['r2'], metric['mre'], metric['mdape']]
        for metric in metrics_dict[0]
    ])
    std_metrics = {
        'mse': np.std(metrics[:, 0]),
        'rmse': np.std(metrics[:, 1]),
        'sse': np.std(metrics[:, 2]),
        'mae': np.std(metrics[:, 3]),
        'r2': np.std(metrics[:, 4]),
        'mre': np.std(metrics[:, 5]),
        'mdape': np.std(metrics[:, 6])
    }
    return std_metrics

def setup_data(seed=42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if HYPERPARAMS['model'] == 'dmpnn' and HYPERPARAMS['chemprop']:
        dataset = FreeSolv(allowed_atoms=allowed_atoms, atom_feature_list=atom_feature_list, bond_feature_list=bond_feature_list)
    else:
        dataset = FreeSolv()
    sample = dataset[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1]

    if HYPERPARAMS['model'] == 'afp':
        train_data, val_data, test_data = dataset.split_and_scale(scale=True, seed=seed, split_type='random', split_frac=[0.8, 0.1, 0.1], is_dmpnn=False)
    elif HYPERPARAMS['model'] == 'dmpnn':
        #dataset = RevIndexedSubSet(dataset)
        train_data, val_data, test_data = dataset.split_and_scale(scale=True, seed=seed, split_type='random', split_frac=[0.8, 0.1, 0.1], is_dmpnn=True)   
        train_data, val_data, test_data = RevIndexedSubSet(train_data), RevIndexedSubSet(val_data), RevIndexedSubSet(test_data)

    HYPERPARAMS['batch_size'] = len(train_data)
    return train_data, val_data, test_data, node_in_dim, edge_in_dim, device

def main():
    all_metrics_bagging = []
    all_metrics_random = []
    all_metrics_jackknife = []
    all_metrics_wbb = []

    for seed in HYPERPARAMS['seed']:
        print(f"\nSeed {seed}:")
        train_data, val_data, test_data, node_in_dim, edge_in_dim, device = setup_data(seed=seed)

        start_time = time.time()
        bagging_metrics = Bagging(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, HYPERPARAMS).run()
        all_metrics_bagging.append(bagging_metrics)
        bagging_time = time.time() - start_time

        start_time = time.time()
        random_metrics = RandomWeightInitialization(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, HYPERPARAMS).run()
        all_metrics_random.append(random_metrics)
        random_weights_time = time.time() - start_time

        start_time = time.time()
        jackknife_metrics = Jackknife(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, HYPERPARAMS).run()
        all_metrics_jackknife.append(jackknife_metrics)
        jackknife_time = time.time() - start_time

        start_time = time.time()
        bb_metrics = BayesianBootstrap(train_data, val_data, test_data, node_in_dim, edge_in_dim, device, HYPERPARAMS).run()
        all_metrics_wbb.append(bb_metrics)
        bb_time = time.time() - start_time

    print("#####################################################################")
    print("Completed with model: ", HYPERPARAMS['model'].upper())

    if HYPERPARAMS['Print Metrics']:
        print("Bagging Metrics:", bagging_metrics)
        print("#####################################################################")
        print_averages(bagging_metrics, 'Bagging')

    print("\nBagging Metrics Std Dev:")
    print(calculate_std_metrics(all_metrics_bagging), "\n")
    print("Bagging Time: ", bagging_time, "\n")

    if HYPERPARAMS['Print Metrics']:
        print_averages(random_metrics, 'Random Initialization')

    print("\nRandom Initialization Metrics Std Dev:")
    print(calculate_std_metrics(all_metrics_random), "\n")
    print("Random Initialization Time: ", random_weights_time, "\n")

    if HYPERPARAMS['Print Metrics']:
        print_averages(jackknife_metrics, 'Jackknife')

    print("\nJackknife Metrics Std Dev:")
    print(calculate_std_metrics(all_metrics_jackknife), "\n")
    print("Jackknife Time: ", jackknife_time, "\n")

    if HYPERPARAMS['Print Metrics']:
        print_averages(bb_metrics, 'Weighted Bayesian Bootstrap')

    print("\nWeighted Bayesian Bootstrap Metrics Std Dev:")
    print(calculate_std_metrics(all_metrics_wbb), "\n")
    print("Weighted Bayesian Bootstrap Time: ", bb_time, "\n")
    print("#####################################################################")

if __name__ == "__main__":
    main()