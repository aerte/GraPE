import mlflow
from typing import Dict


def setup_run_name(config: Dict):
    run_name = config['run_name']
    model_name = config['model_name']
    print(f"Model name: {model_name}")
    if 'dmpnn' in model_name.lower():
        config['run_name'] = f"DMPNN_{run_name}"
        print(f"Run name: {config['run_name']}")
    elif 'afp' in model_name.lower():
        config['run_name'] = f"AFP_{run_name}"
        print(f"Run name: {config['run_name']}")

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