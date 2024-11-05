import mlflow
from typing import Dict
import pandas as pd
import os
import datetime

from grape_chem.utils.data import save_splits_to_csv

def setup_run_name(config: Dict):
    time = datetime.datetime.now().strftime("%d-%m-%Y-%H")
    run_name = config.get('run_name')
    run_name = run_name + f"_{time}" if run_name else time

    # Add data name to run_name if 'data_path' exists
    if len(config['data_labels']) > 1:
        data_name = "_".join(config['data_labels'])
    elif len(config['data_labels']) == 1:
        data_name = config['data_labels'][0] 
    else:
        ValueError("No data labels found in config['data_labels']")
    
    if data_name:
        run_name = f"{data_name}_{run_name}"

    # Determine model name
    model_name = config['model_name'].lower()
    model_name = model_name + f"_{time}" if model_name else time
    config['model_name'] = model_name


    # Update the run_name based on the model type
    if 'originaldmpnn' in model_name:
        run_name = f"OriginalDMPNN_{run_name}"
    elif 'dmpnn' in model_name:
        run_name = f"DMPNN_{run_name}"
    elif 'afp' in model_name:
        run_name = f"AFP_{run_name}"
    else:
        run_name = f"Custom_{run_name}"

    # Update and print run_name in config
    config['run_name'] = run_name


def setup_mlflow(config:Dict):
    # Very important to set the tracking URI to the server's URI inside the function where the expertiment is set. 
    # Otherwise, it will not work. See: >>>> https://github.com/mlflow/mlflow/issues/3729 <<<<
    setup_run_name(config)
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(config['experiment_name'])
    
    print(f"Current experiment: {mlflow.get_experiment_by_name(config['experiment_name'])}")
    return mlflow.start_run(run_name=config['run_name'])

def log_params_to_mlflow(params):
    mlflow.log_params(params)

def log_latents(latents, file_path, log_latent_batch_index=0):
    # Convert latents to NumPy for logging
    latents_to_log = latents.cpu().detach().numpy()  # Ensure it's on CPU and converted to NumPy
    latent_file_path = file_path + 'latents.csv'
    
    # Save the latents to a CSV file
    pd.DataFrame(latents_to_log).to_csv(latent_file_path, index=False)

    if mlflow.active_run():
        # Log the latent representation as an artifact
        mlflow.log_artifact(latent_file_path)

def track_params(config: Dict, data_bundle: Dict):
    """Log hyperparameters and dataset information to MLflow."""
    if mlflow.active_run():
        mlflow.log_params(config)

        df, train_data, val_data, test_data, data = (
            data_bundle['df'], data_bundle['train_data'], 
            data_bundle['val_data'], data_bundle['test_data'], 
            data_bundle['data']
        )
        index = len(data)//2 if (len(data)) % 2 == 0 else (len(data))//2 + 1
        mlflow.log_param("Target", config['target'])
        mlflow.log_param("Global Features", config['global_features'])
        mlflow.log_param("Learning Rate", config['learning_rate'])
        mlflow.log_param("smile_example", data.smiles[index])
        mlflow.log_param("atom_example", data[index].x)
        mlflow.log_param("bond_example", data[index].edge_attr)
        mlflow.log_params({
            "dataset_length": len(data),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data)
        })

        if config['save_splits']:
            print("Saving dataset splits now...")
            mlflow.log_artifact(save_splits_to_csv(df, train_data, val_data, test_data, config['save_path']))
    
    else:
        print("No active MLflow run. Skipping logging of hyperparameters and dataset information.") 