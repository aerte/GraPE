# Original code. Refactored into Grape.utils.ensemble and a ensemble notebook to showcase
import torch
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from torch_geometric.data import Data
from grape_chem.models import AFP, DMPNN
from grape_chem.datasets import FreeSolv
from grape_chem.utils import train_model, test_model, pred_metric, RevIndexedSubSet
from grape_chem.utils.model_utils import set_seed
from grape_chem.utils import EarlyStopping
from sklearn.utils import resample
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import time
import random

# Hyperparameters
HYPERPARAMS = {
    'learning_rate': 0.002,
    'batch_size': None,  # Set in setup_data
    'epochs': 200,
    'n_models': 10,
    'early_stopping_patience': 50,
    'scheduler_patience': 5,
    'scheduler_factor': 0.9,
    'seed': [42],
    'weight_seed': 100,
    'Print Metrics': True,
    'model': 'afp',  # 'afp' or 'dmpnn'
    'chemprop': True
}

# Chemprop allowed atoms and features
allowed_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'I']
atom_feature_list = ['chemprop_v2_atom_features']
bond_feature_list = ['chemprop_v2_bond_features']

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
    # Convert the list of dictionaries into a list of lists
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

# Function to calculate molecular weight
def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

# Define the dataset and split
def setup_data(seed=42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if HYPERPARAMS['model'] == 'dmpnn' and HYPERPARAMS['chemprop']:
        #Missing molecular weight feature as global feature
        dataset = FreeSolv(allowed_atoms=allowed_atoms, atom_feature_list=atom_feature_list, bond_feature_list=bond_feature_list)
    else:
        dataset = FreeSolv()
    sample = dataset[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1]

    if HYPERPARAMS['model'] == 'afp':
        train_data, val_data, test_data = dataset.split_and_scale(scale=True, seed=seed, split_type='random', split_frac=[0.8, 0.1, 0.1], is_dmpnn=False)

    elif HYPERPARAMS['model'] == 'dmpnn':
        train_data, val_data, test_data = dataset.split_and_scale(scale=True, seed=seed, split_type='random', split_frac=[0.8, 0.1, 0.1], is_dmpnn=True)   
        train_data, val_data, test_data = RevIndexedSubSet(train_data), RevIndexedSubSet(val_data), RevIndexedSubSet(test_data)

    # Set the batch size for training and evaluation
    HYPERPARAMS['batch_size'] = len(train_data)
    return train_data, val_data, test_data, node_in_dim, edge_in_dim, device

def predict_ensemble(models, device, test_data):
    predictions = []
    metrics = []
    for model in models:
        model.to(device)
        pred = test_model(model=model, test_data_loader=test_data, device=device, batch_size=len(test_data))
        predictions.append(pred)
        test_targets = [data.y for data in test_data]
        targets = torch.cat(test_targets, dim=0).to(device)
        metric_results = pred_metric(prediction=pred, target=targets, metrics='all', print_out=False)
        metrics.append(metric_results)
    return metrics

# Bagging Technique
def bagging(train_data, val_data, test_data, node_in_dim, edge_in_dim, device):
    def create_bootstrap_samples(data, n_samples):
        samples = []
        for _ in range(n_samples):
            indices = resample(range(len(data)), replace=True)
            sample_data = [data[i] for i in indices]
            samples.append(sample_data)
        return samples

    def train_models(train_samples, val_samples, n_models):
        models = []
        for i, sample_data in enumerate(train_samples):
            print(f'Training Bagging model {i + 1}/{n_models}')
            val_data = val_samples[i]
            if HYPERPARAMS['model'] == 'afp':
                model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
            elif HYPERPARAMS['model'] == 'dmpnn':
                model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
            else:
                raise ValueError('Invalid model type')
            criterion = MSELoss()
            optimizer = Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
            early_stopping = EarlyStopping(patience=HYPERPARAMS['early_stopping_patience'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=HYPERPARAMS['scheduler_factor'], patience=HYPERPARAMS['scheduler_patience']
            )

            train_model(model=model, loss_func=criterion, optimizer=optimizer, scheduler=scheduler,
                        train_data_loader=sample_data, val_data_loader=val_data, epochs=HYPERPARAMS['epochs'], device=device,
                        batch_size=HYPERPARAMS['batch_size'], early_stopper=early_stopping)
            model.load_state_dict(torch.load('best_model.pt'))
            models.append(model)
        return models

    train_samples = create_bootstrap_samples(train_data, HYPERPARAMS['n_models'])
    val_samples = create_bootstrap_samples(val_data, HYPERPARAMS['n_models'])
    models = train_models(train_samples, val_samples, HYPERPARAMS['n_models'])
    metrics = predict_ensemble(models, device, test_data)
    return metrics

# Random Weight Initialization Technique
def random_weight_initialization(train_data, val_data, test_data, node_in_dim, edge_in_dim, device):
    def train_models(n_models):
        models = []
        for i in range(n_models):
            print(f'Training Random Initialization model {i + 1}/{n_models}')
            set_seed(HYPERPARAMS['weight_seed'] + i)
            if HYPERPARAMS['model'] == 'afp':
                model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
            elif HYPERPARAMS['model'] == 'dmpnn':
                model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
            else:
                raise ValueError('Invalid model type')
            criterion = MSELoss()
            optimizer = Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
            early_stopping = EarlyStopping(patience=HYPERPARAMS['early_stopping_patience'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=HYPERPARAMS['scheduler_factor'], patience=10
            )

            train_model(model=model, loss_func=criterion, optimizer=optimizer, scheduler=scheduler,
                        train_data_loader=train_data, val_data_loader=val_data, epochs=HYPERPARAMS['epochs'], device=device,
                        batch_size=HYPERPARAMS['batch_size'], early_stopper=early_stopping)
            model.load_state_dict(torch.load('best_model.pt'))
            models.append(model)
        return models

    models = train_models(HYPERPARAMS['n_models'])
    metrics = predict_ensemble(models, device, test_data)
    return metrics

# Jackknife Technique
def jackknife(train_data, val_data, test_data, node_in_dim, edge_in_dim, device):
    def create_synth_samples(reference_model, data, n_samples):
        synth_samples = []
        reference_model = reference_model.to(device)
        with torch.no_grad():
            predictions = test_model(reference_model, test_data_loader=data, device=device, batch_size=len(data))
        actual_values = [data[i].y.to(device) for i in range(len(data))]
        residuals = [actual_values[i] - predictions[i] for i in range(len(data))]

        for _ in range(n_samples):
            sampled_residuals = torch.stack(resample(residuals, replace=True))
            synthetic_values = predictions + sampled_residuals[0]
            synthetic_dataset = [Data(x=data[i].x, edge_index=data[i].edge_index, edge_attr=data[i].edge_attr,
                                      y=synthetic_values[i], revedge_index=getattr(data[i], 'revedge_index', None)) for i in range(len(data))]
            synth_samples.append(synthetic_dataset)
        return synth_samples

    def train_models(jackknife_samples):
        jackknife_models = []
        criterion = MSELoss()
        for i, synthetic_data in enumerate(jackknife_samples):
            print(f'Training Jackknife model {i + 1}/{len(jackknife_samples)}')
            if HYPERPARAMS['model'] == 'afp':
                model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
            elif HYPERPARAMS['model'] == 'dmpnn':
                model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
            else:
                raise ValueError('Invalid model type')
            optimizer = Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
            early_stopping = EarlyStopping(patience=HYPERPARAMS['early_stopping_patience'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=HYPERPARAMS['scheduler_factor'], patience=10
            )

            train_model(model=model, loss_func=criterion, optimizer=optimizer, scheduler=scheduler,
                        train_data_loader=synthetic_data, val_data_loader=val_data, epochs=HYPERPARAMS['epochs'], device=device,
                        batch_size=len(synthetic_data), early_stopper=early_stopping)
            model.load_state_dict(torch.load('best_model.pt'))
            jackknife_models.append(model)
        return jackknife_models

    if HYPERPARAMS['model'] == 'afp':
        reference_model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
    elif HYPERPARAMS['model'] == 'dmpnn':
        reference_model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
    criterion = MSELoss()
    optimizer = Adam(reference_model.parameters(), lr=HYPERPARAMS['learning_rate'])
    early_stopping = EarlyStopping(patience=HYPERPARAMS['early_stopping_patience'])
    train_model(model=reference_model, loss_func=criterion, optimizer=optimizer, train_data_loader=train_data,
                val_data_loader=val_data, epochs=HYPERPARAMS['epochs'], device=device, batch_size=HYPERPARAMS['batch_size'], early_stopper=early_stopping)
    jackknife_samples = create_synth_samples(reference_model, train_data, HYPERPARAMS['n_models'])
    jackknife_models = train_models(jackknife_samples)
    metrics = predict_ensemble(jackknife_models, device, test_data)
    return metrics

# Bayesian Bootstrap Technique
def bayesian_bootstrap(train_data, val_data, test_data, node_in_dim, edge_in_dim, device):
    def Dirichlet_sample(m, n):
        """Returns a matrix of Dirichlet-distributed weights."""
        Dirichlet_params = np.ones(m * n)
        Dirichlet_weights = np.asarray([np.random.gamma(a, 1) for a in Dirichlet_params])
        Dirichlet_weights = Dirichlet_weights.reshape(m, n)
        row_sums = Dirichlet_weights.sum(axis=1)
        Dirichlet_weights /= row_sums[:, np.newaxis]
        return Dirichlet_weights

    def create_bayesian_samples(data, n_samples):
        weights_matrix = Dirichlet_sample(n_samples, len(data))
        samples = []
        
        for weights in weights_matrix:
            indices = np.random.choice(len(data), len(data), replace=True, p=weights)
            sampled_data = [data[i] for i in indices]
            samples.append(sampled_data)
        
        return samples

    def train_models(train_samples, val_samples, n_models):
        models = []
        for i, sample_data in enumerate(train_samples):
            print(f'Training Bayesian Bootstrap model {i + 1}/{n_models}')
            val_data = val_samples[i]
            if HYPERPARAMS['model'] == 'afp':
                model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
            elif HYPERPARAMS['model'] == 'dmpnn':
                model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
            else:
                raise ValueError('Invalid model type')
            
            criterion = MSELoss()
            optimizer = Adam(model.parameters(), lr=HYPERPARAMS['learning_rate'])
            early_stopping = EarlyStopping(patience=HYPERPARAMS['early_stopping_patience'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=HYPERPARAMS['scheduler_factor'], patience=HYPERPARAMS['scheduler_patience']
            )

            train_model(model=model, loss_func=criterion, optimizer=optimizer, scheduler=scheduler,
                        train_data_loader=sample_data, val_data_loader=val_data, epochs=HYPERPARAMS['epochs'], device=device,
                        batch_size=HYPERPARAMS['batch_size'], early_stopper=early_stopping)
            model.load_state_dict(torch.load('best_model.pt'))
            models.append(model)
        return models

    # Generate bootstrap samples
    train_samples = create_bayesian_samples(train_data, HYPERPARAMS['n_models'])
    val_samples = create_bayesian_samples(val_data, HYPERPARAMS['n_models'])

    # Train models
    models = train_models(train_samples, val_samples, HYPERPARAMS['n_models'])
    
    # Evaluate models
    metrics = predict_ensemble(models, device, test_data)
    return metrics

def main():
    all_metrics_bagging = []
    all_metrics_random = []
    all_metrics_jackknife = []
    all_metrics_wbb = []

    for seed in HYPERPARAMS['seed']:
        print(f"\nSeed {seed}:")
        train_data, val_data, test_data, node_in_dim, edge_in_dim, device = setup_data(seed=seed)

        start_time = time.time()
        bagging_metrics = bagging(train_data, val_data, test_data, node_in_dim, edge_in_dim, device)
        all_metrics_bagging.append(bagging_metrics)
        bagging_time = time.time() - start_time

        start_time = time.time()
        random_metrics = random_weight_initialization(train_data, val_data, test_data, node_in_dim, edge_in_dim, device)
        all_metrics_random.append(random_metrics)
        random_weights_time = time.time() - start_time

        start_time = time.time()
        jackknife_metrics = jackknife(train_data, val_data, test_data, node_in_dim, edge_in_dim, device)
        all_metrics_jackknife.append(jackknife_metrics)
        jackknife_time = time.time() - start_time

        start_time = time.time()
        bb_metrics = bayesian_bootstrap(train_data, val_data, test_data, node_in_dim, edge_in_dim, device)
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
