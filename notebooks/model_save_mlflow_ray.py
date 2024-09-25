import torch
import pandas as pd
import numpy as np
import datetime
from grape_chem.models import DMPNN, AFP
from grape_chem.utils import train_model, test_model, pred_metric, return_hidden_layers, RevIndexedSubSet, DataSet
from grape_chem.utils.model_utils import set_seed, create_checkpoint
from grape_chem.utils.featurizer import AtomFeaturizer
from grape_chem.utils import EarlyStopping

from torch.optim import lr_scheduler
from rdkit import Chem
from rdkit.Chem import Descriptors

import mlflow
import mlflow.pytorch

import os
from typing import Dict
from itertools import product

# Ray Tune imports
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Function to create paths in a cross-platform manner
def get_path(*args):
    return os.path.normpath(os.path.join(*args))

# Absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Shorten directory name to avoid exceeding path length limits
def custom_trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

def get_model_dir(model_name):
    model_dirs = {
        'afp': get_path(current_dir, '../models', 'AFP'),
        'dmpnn': get_path(current_dir, '../models', 'DMPNN')
    }

    model_type = next((k for k in model_dirs.keys() if k in model_name.lower()), None)

    if model_type is None:
        raise ValueError(f"Model name {model_name} not recognized") 

    return model_dirs[model_type]  

def train_model_experiment(config: Dict):
    # Very important to set the tracking URI to the server's URI inside the function where the expertiment is set. 
    # Otherwise, it will not work. See: >>>> https://github.com/mlflow/mlflow/issues/3729 <<<<
    mlflow.set_tracking_uri('http://localhost:5000') 
    mlflow.set_experiment(config['experiment_name'])
    print(f"Current experiment: {mlflow.get_experiment_by_name(config['experiment_name'])}")

    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params(config)
        
        set_seed(config['model_seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        df = pd.read_csv(config['data_path'], sep=';', encoding='latin')
        smiles = df['SMILES']
        target = df['Const_Value']
        molecular_weights = np.array([calculate_molecular_weight(s) for s in smiles])
        global_features = np.array(molecular_weights)
        
        mlflow.log_param("num_smiles", len(smiles))
        
        data = DataSet(smiles=smiles, target=target, global_features=global_features, 
                       allowed_atoms=config['allowed_atoms'], 
                       atom_feature_list=config['atom_feature_list'], 
                       bond_feature_list=config['bond_feature_list'], 
                       log=False, only_organic=False, filter=True, allow_dupes=True)
        
        sample = data[50]
        node_in_dim = sample.x.shape[1]
        edge_in_dim = sample.edge_attr.shape[1]
        mlp = return_hidden_layers(config['mlp_layers'])
        
        mlflow.log_param("dataset_length", len(data))
        
        train_data, val_data, test_data = data.split_and_scale(
            split_frac=[0.8, 0.1, 0.1], scale=True, seed=config['model_seed'], 
            is_dmpnn='dmpnn' in config['model_name'], split_type='random'
        )

        mlflow.log_params({
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data)
        })

        dataset_dict = {
            'allowed_atoms': data.allowed_atoms, 
            'atom_feature_list': data.atom_feature_list, 
            'bond_feature_list': data.bond_feature_list, 
            'data_mean': data.mean, 
            'data_std': data.std, 
            'data_name': data.data_name
        }
        
        # Initialize model
        if 'dmpnn' in config['model_name'].lower():
            model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, node_hidden_dim=config['hidden_dim'], 
                          dropout=config['dropout'], mlp_out_hidden=mlp, dataset_dict=dataset_dict)
        if 'afp' in config['model_name'].lower():
            model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, out_dim=1, dataset_dict=dataset_dict)
        
        mlflow.log_params({
            "model": config['model_name'],
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "out_dim": 1
        })
        
        num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("num_learnable_params", num_learnable_params)
        
        model = model.to(device)
        
        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        early_stopper = EarlyStopping(patience=config['patience'], model_name=config['model_name'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9999, min_lr=0.0000000000001,
                                                   patience=config['patience_scheduler'])
        
        loss_func = torch.nn.functional.l1_loss
        
        # Log metrics during training
        train_loss, val_loss = train_model(model=model, loss_func=loss_func, optimizer=optimizer, 
                    train_data_loader=train_data, val_data_loader=val_data, 
                    epochs=config['epochs'], batch_size=config['batch_size'], 
                    early_stopper=early_stopper, scheduler=scheduler, device=device)
        
        mlflow.log_param("train_loss", train_loss)
        mlflow.log_param("val_loss", val_loss)

        pred = test_model(model=model, test_data_loader=test_data, device=device, batch_size=len(test_data))
        test_targets = [data.y for data in test_data]
        targets = torch.cat(test_targets, dim=0).to(device)
        test_loss = torch.nn.functional.l1_loss(pred, targets)
        mlflow.log_param("test_loss", test_loss)

        # Report the best validation loss to Ray Tune
        #print("All validation losses: ", val_loss)
        best_val_loss = min(val_loss)
        epoch = val_loss.index(best_val_loss)
        metrics = {"val_loss": best_val_loss}
        #print(f"Best validation loss: {best_val_loss}")
        train.report(metrics)

        # Log the final trained model
        mlflow.pytorch.log_model(model, "model", pip_requirements=config['pip_requirements'])
        
        # Save model locally
        model_dir = get_model_dir(config['model_name'])
        model_path = os.path.join(model_dir, f"{config['model_name']}_final.pt")
        torch.save(create_checkpoint(model), model_path)
        mlflow.log_artifact(model_path)
        # Create list of 0 to len(train_loss) for plotting
        epoch_plot = list(range(len(train_loss)))
        mlflow.log_param("epoch_plot", epoch_plot)
        
        print(f"Model saved as {config['model_name']}.pt in {model_dir}")

def main():
    set_seed(1)

    # Define configurations for Ray Tune
    time = datetime.datetime.now().strftime("%d-%m-%Y-%H")
    config = {
        'experiment_name': "Ray Molecular Property Prediction Training",  # Added this line
        'run_name': f"Model Training - {time}",  # Added this line
        'epochs': tune.choice([50, 100]),
        'batch_size': tune.choice([None]),  # None means full batch
        'hidden_dim': tune.choice([300]),
        'dropout': tune.uniform(0.0, 0.5),
        'patience': tune.choice([20]),
        'patience_scheduler': tune.choice([5]),
        'learning_rate': tune.loguniform(1e-4, 1e-2),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        'model_seed': tune.randint(1, 100),
        'mlp_layers': tune.choice([4]),
        'model_name': tune.choice([f'{time}_afp', f'{time}_dmpnn']),  # Changed to tune.choice
        'data_path': get_path(current_dir, '../data/dippr.csv'),
        'allowed_atoms': set(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'I']),
        'atom_feature_list': ['chemprop_atom_features'],
        'bond_feature_list': ['chemprop_bond_features'],
        'pip_requirements': [get_path(current_dir, '../requirements.txt')]
    }

    # Setup Ray Tune scheduler
    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=10,
        reduction_factor=2)

    # Run the tuning process
    analysis = tune.run(
        tune.with_parameters(train_model_experiment),  
        metric="val_loss", # metric to optimize
        mode="min", # minimize the metric
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        storage_path= get_path(current_dir, '../ray_results'), # Directory to save results
        trial_dirname_creator=custom_trial_dirname_creator,  # Add this to shorten the trial directory names
    )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    main()
