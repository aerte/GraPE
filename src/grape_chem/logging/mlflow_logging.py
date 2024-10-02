import mlflow
from typing import Dict
import pandas as pd
import os


def setup_run_name(config: Dict):
    run_name = config.get('run_name', '')

    # Add data name to run_name if 'data_path' exists
    data_name = os.path.splitext(os.path.basename(config.get('data_path', '')))[0]
    if data_name:
        run_name = f"{data_name}_{run_name}"

    # Determine model name
    model_name = config.get('model_name') or config.get('model_class', '').lower()

    # Update the run_name based on the model type
    if 'dmpnn' in model_name:
        run_name = f"DMPNN_{run_name}"
    elif 'afp' in model_name:
        run_name = f"AFP_{run_name}"

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