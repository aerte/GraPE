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

import json
import os
from typing import Dict
from itertools import product

def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

def get_model_dir(model_name):
    # Add other models such as Megnet and MPNN
    model_dirs = {'afp': 'models\\AFP', 'dmpnn': 'models\\DMPNN'}

    model_type = next((k for k in model_dirs.keys() if k in model_name.lower()), None)

    if model_type is None:
        raise ValueError(f"Model name {model_name} not recognized") 

    return model_dirs[model_type]  

def train_model_experiment(config: Dict):
    
    mlflow.set_experiment(config['experiment_name'])
    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params(config)
        
        set_seed(config['model_seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load and prepare data
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
            is_dmpnn=False, split_type='random'
        )
        
        mlflow.log_params({
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data)
        })
        
        if 'dmpnn' in config['model_name']:
            train_data, val_data, test_data = RevIndexedSubSet(train_data), RevIndexedSubSet(val_data), RevIndexedSubSet(test_data)
        
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
        
        train_model(model=model, loss_func=loss_func, optimizer=optimizer, 
                    train_data_loader=train_data, val_data_loader=val_data, 
                    epochs=config['epochs'], batch_size=config['batch_size'], 
                    early_stopper=early_stopper, scheduler=scheduler, device=device)
        
        # Log the final trained model
        mlflow.pytorch.log_model(model, "model", pip_requirements=config['pip_requirements'])
        
        # Save model locally
        model_dir = get_model_dir(config['model_name'])
        model_path = os.path.join(model_dir, f"{config['model_name']}.pt")
        torch.save(create_checkpoint(model), model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Model saved as {config['model_name']}.pt in {model_dir}")

def main():
    mlflow.set_tracking_uri('http://localhost:5000')
    # Define configurations
    time = datetime.datetime.now().strftime("%d-%m-%Y-%H")
    configs = {
        'experiment_name': ["Molecular Property Prediction Training"],
        'run_name': ["Model Training"],
        'epochs': [1],
        'batch_size': [32],
        'hidden_dim': [300],
        'depth': [3],
        'dropout': [0.0],
        'patience': [4],
        'patience_scheduler': [5],
        'warmup_epochs': [2],
        'learning_rate': [0.0001],
        'weight_decay': [0.0],
        'model_seed': [42],
        'mlp_layers': [4],
        'atom_layers': [3],
        'mol_layers': [3],
        'model_name': [f'{time}_afp', f'{time}_dmpnn'],
        'data_path': ['C:\\Users\\Thoma\\GraPE\\notebooks\\dippr.csv'],
        'allowed_atoms': [['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'I']],
        'atom_feature_list': [['chemprop_atom_features']],
        'bond_feature_list': [['chemprop_bond_features']],
        'pip_requirements': ["C:\\Users\\Thoma\\code\\GraPE\\requirements.txt"]
    }
    
    # Generate all possible combinations of configurations
    keys, values = zip(*configs.items())
    configurations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Run experiments for each configuration
    for i, config in enumerate(configurations):
        print(f"Running training experiment {i+1}/{len(configurations)}")
        config['run_name'] = f"{config['run_name']} - Config {i+1}"
        train_model_experiment(config)

if __name__ == "__main__":
    main()