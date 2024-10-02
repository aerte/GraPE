import torch
import datetime

# new functions
# load_model_from_data, load_data_from_csv, parity_plot_mlflow, setup_mlflow, 
# log_latents, load_config, get_path, get_model_dir, hyperparameter_tuning
from grape_chem.utils import train_model, test_model, pred_metric
from grape_chem.utils.data import load_data_from_csv, get_path, get_model_dir
from grape_chem.utils.model_utils import set_seed, create_checkpoint, load_model_from_data
from grape_chem.plots.post_plots import parity_plot, parity_plot_mlflow
from grape_chem.utils import EarlyStopping
from grape_chem.logging.hyperparameter_tuning import hyperparameter_tuning
from grape_chem.logging.mlflow_logging import setup_mlflow, log_latents
from grape_chem.logging.config import load_config

from torch.optim import lr_scheduler
from rdkit import Chem
from rdkit.Chem import Descriptors

import mlflow
import mlflow.pytorch

import os
from typing import Dict

# Ray Tune imports
from ray import train
from ray import tune
## Start server with: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

def train_model_experiment(config: Dict):
    model_dir = get_model_dir(current_dir, config['model_name'])

    config['save_path'] = model_dir
    with setup_mlflow(config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        mlflow.log_params(config)
        # Setting seed and data_setup
        set_seed(config['model_seed'])
        # Set limit to 1000 for testing purposes
        train_data, val_data, test_data, data = load_data_from_csv(config, return_dataset=True, limit = 1000)
        #train_data, val_data, test_data, data = load_data_from_csv(config, return_dataset=True)
        # Load the model
        model = load_model_from_data(data, config, device)
        mlflow.pytorch.log_model(model, artifact_path="model")
        
        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        early_stopper = EarlyStopping(patience=config['patience'], model_name=config['model_name'], save_path=config['save_path'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=0.0000000000001,
                                                   patience=config['patience_scheduler'])
        
        loss_func = torch.nn.functional.l1_loss
        
        # train_model. Log metrics during training
        train_loss, val_loss = train_model(model=model, loss_func=loss_func, optimizer=optimizer, 
                    train_data_loader=train_data, val_data_loader=val_data, 
                    epochs=config['epochs'], batch_size=config['batch_size'], 
                    early_stopper=early_stopper, scheduler=scheduler, device=device)
        
        mlflow.log_param("train_loss_after_train", train_loss)
        mlflow.log_param("val_loss_after_train", val_loss)

        # mlflow boolean to log mlflow metrics
        pred, latents = test_model(model=model, test_data_loader=test_data, device=device, batch_size=len(test_data), return_latents=True, mlflow_log=True)
        test_targets = [data.y for data in test_data]
        
        targets = torch.cat(test_targets, dim=0).to(device)
        test_loss = torch.nn.functional.l1_loss(pred, targets)
        mlflow.log_param("test_loss_after_test", test_loss)

        # RAYTUNE ### Report the best validation loss to Ray Tune ###
        #print("All validation losses: ", val_loss)
        best_val_loss = min(val_loss)
        best_epoch = val_loss.index(best_val_loss)
        mlflow.log_param("best_epoch", best_epoch)
        metrics = {"val_loss": best_val_loss}
        mlflow.log_metric("best_val_loss", best_val_loss)

        # RAYTUNE ### Report the best validation loss to Ray Tune ###
        train.report(metrics)

        # Log the final trained model
        mlflow.pytorch.log_model(model, "model", pip_requirements=config['pip_requirements'])
        
        # Save model locally
        model_dir = config['save_path']
        print(f"Model saved in {model_dir}")
        model_path = os.path.join(model_dir, f"{config['model_name']}_final.pt")
        torch.save(create_checkpoint(model), model_path)
        mlflow.log_param("Model_path", model_path)

        # Log the latents as an artifact
        log_latents(latents, model_dir, best_epoch)

        # Log the parity plot as an artifact
        mlflow.log_artifact(parity_plot_mlflow("test predictions vs targets", targets.cpu().detach().numpy(), pred.cpu().detach().numpy(), config['save_path']))
        parity_plot(pred.cpu().detach().numpy(), targets.cpu().detach().numpy(), path_to_export=config['save_path'])
        
        print(f"Model saved as {config['model_name']}.pt in {model_dir}")

def main():
    set_seed(1)

    # Load config file from its path, in this case its in the same directory as the script
    path = get_path(current_dir, 'config.yaml')
    base_config = load_config(path)
    # Important to set the paths in the config file
    
    # Define configurations for Ray Tune
    time = datetime.datetime.now().strftime("%d-%m-%Y-%H")
    search_space = {
        'epochs': tune.choice([300]),
        'batch_size': tune.choice([4192]),#tune.choice([1048,4192,8384]), # None for full dataset
        'model_name': tune.choice([f'DMPNN_{time}']), #,}'
        'hidden_dim': tune.choice([300]),
        'dropout': tune.uniform(0.0, 0.1),
        'patience': tune.choice([50]),
        'patience_scheduler': tune.choice([10]),
        'learning_rate': tune.loguniform(1e-3, 1e-2),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        'model_seed': tune.choice([42]),
        'mlp_layers': tune.choice([4]),
        #'model_name': tune.choice([f'DMPNN_{time}']),
        'run_name': f"Model Training - {time}"
    }
    config = {**base_config, **search_space}

    hyperparameter_tuning(train_model_experiment, config, num_samples=1, storage_path=get_path(current_dir, '../ray_results'))

if __name__ == "__main__":
    main()
