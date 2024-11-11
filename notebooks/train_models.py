import torch
import datetime
import numpy as np
import os
from typing import Dict

import matplotlib.pyplot as plt
# new functions
# load_model_from_data, load_dataset_from_csv, parity_plot_mlflow, setup_mlflow, 
# log_latents, load_config, get_path, get_model_dir, hyperparameter_tuning
# load_best_model, evaluate_model, log_and_report_metrics

from grape_chem.utils import train_model
from grape_chem.utils.data import load_dataset_from_csv, get_path, get_model_dir
from grape_chem.utils.model_utils import set_seed, create_checkpoint, load_model_from_data, load_best_model, evaluate_model_mlflow
from grape_chem.utils import EarlyStopping
from grape_chem.logging.hyperparameter_tuning import hyperparameter_tuning, log_and_report_metrics
from grape_chem.logging.mlflow_logging import setup_mlflow, track_params
from grape_chem.logging.config import load_config

from torch.optim import lr_scheduler

import mlflow
import mlflow.pytorch

from ray import tune

# Start MLflow server with the following command:
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

current_dir = os.path.dirname(os.path.abspath(__file__))

# Model training functions
def initialize_model_and_optimizer(config, data, device):
    """Initialize the model and optimizer."""
    model = load_model_from_data(data, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    return model, optimizer

def setup_training_environment(config: Dict, optimizer):
    """Set up the training environment and return the scheduler and early stopper."""
    early_stopper = EarlyStopping(patience=config['patience'], model_name=config['model_name'], save_path=config['model_path'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['schedule_factor'], 
                                                min_lr=1.00E-09, patience=config['patience_scheduler'])
    return early_stopper, scheduler

def train_and_validate_model(model, optimizer, config, train_data, val_data, early_stopper, scheduler, device):
    """Train the model and validate it on the validation set."""
    loss_func = torch.nn.functional.l1_loss
    train_loss, val_loss = train_model(
        model=model, loss_func=loss_func, optimizer=optimizer, 
        train_data_loader=train_data, val_data_loader=val_data, 
        epochs=config['epochs'], batch_size=config['batch_size'], 
        early_stopper=early_stopper, scheduler=scheduler, device=device
    )
    return train_loss, val_loss

# Plotting and logging functions
def create_barplot(metrics_by_type, config):
    """Plot and log metrics for multiple targets."""
    for metric, values in metrics_by_type.items():
        plt.figure(figsize=(8, 6))
        plt.bar(config['data_labels'], values, color=['skyblue', 'lightgreen', 'salmon'], data=values)
        plt.xlabel('Target')
        plt.ylabel(f'{metric.upper()} Value')
        plt.title(f'{metric.upper()} Comparison Across Targets')
        plt.tight_layout()
        for i in range(len(values)):
            plt.text(i,values[i]/2,values[i], ha = 'center',
                bbox = dict(facecolor = 'white', alpha = .5))

        plot_filename = f'{metric}_comparison_plot.png'
        plt.savefig(plot_filename)
        plt.close()

        if mlflow.active_run():
            mlflow.log_artifact(plot_filename)

def save_final_model(model, config):
    """Save the final model and log its path to MLflow."""
    model_path = os.path.join(config['model_path'], f"{config['model_name']}_final.pt")
    mlflow.log_param("Model_path", model_path)

    torch.save(create_checkpoint(model), model_path)
    print(f"Final model saved: {model_path}")

def log_duration(start_time):
    """Log the duration of the training process to MLflow."""
    end_time = datetime.datetime.now()
    mlflow.log_param("duration", end_time - start_time)


# Main function for training the model
def run_experiment(config: Dict, data_bundle: Dict):
    """
    Main function to train the model, log results, and manage the training workflow.

    This function orchestrates the end-to-end model training process, from setting up 
    the environment and initializing the model to logging results in MLflow. The function
    uses early stopping to load the best model, logs evaluation metrics, and saves the final model.

    Args:
        config (Dict): Configuration dictionary containing essential settings for the experiment, including:
            - 'seed' (int): Random seed for reproducibility.
            - 'data_files' (List): List of data files to load.
            - 'data_labels' (List): List of target labels. Used to determine the number of targets.
            - 'model_name' (str): Name for saving model files.
            - 'batch_size' (int): Batch size for training and evaluation.
            - 'model_path' (str): Directory path to save models.
            - Additional settings specific to model architecture, optimizer, and scheduler.
        data_bundle (Dict): Dictionary containing data and data loaders required for training and testing. Expected keys:
            - 'train_data' (DataLoader or Dataset): Training dataset or DataLoader instance.
            - 'val_data' (DataLoader or Dataset): Validation dataset or DataLoader instance.
            - 'test_data' (DataLoader or Dataset): Test dataset or DataLoader instance for final evaluation.
            - 'data' (object): Data object with mean and std attributes for normalization.

    Output:
        None: All logging and model saving occurs within the function.

    Process Outline:
        1. **MLflow Setup**: Establishes an MLflow experiment context for logging parameters, metrics, and models.
        2. **Seed Setting**: Ensures reproducibility by setting the random seed from the `config`.
        3. **Device Selection**: Chooses CUDA or CPU for model computation based on availability.
        4. **Data Extraction**: Extracts training, validation, test data, and normalization information from `data_bundle`.
        5. **Model Directory Setup**: Configures the directory path to save model checkpoints.
        6. **Model and Optimizer Initialization**: Initializes model and optimizer based on `config` specifications.
        7. **MLflow Model Logging**: Logs the initialized model structure to MLflow.
        8. **Training Environment Setup**: Configures early stopping and scheduler for training management.
        9. **Model Training**: Calls `train_and_validate_model` to train and validate the model over epochs.
       10. **Validation Metrics Logging**: Logs validation loss and other metrics.
       11. **Best Model Loading**: Uses early stopping to load the best model checkpoint.
       12. **Model Evaluation**: Evaluates the model on the test set, logging performance metrics and creating parity plots.

    Example Usage:
        run_experiment(config=config, data_bundle=data_bundle)
    """
    #with setup_mlflow(config):
    start_time = datetime.datetime.now()
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Extract data
    train_data, val_data, test_data, data = (
        data_bundle['train_data'], data_bundle['val_data'], 
        data_bundle['test_data'], data_bundle['data']
    )
    config['model_path'] = get_model_dir(current_dir, model_name=config['model_name'])

    # Track parameters and initialize model
    track_params(config, data_bundle)
    model, optimizer = initialize_model_and_optimizer(config, data, device)
    #mlflow.pytorch.log_model(model, artifact_path="model")

    # Set up training environment
    early_stopper, scheduler = setup_training_environment(config, optimizer)

    # Train and validate the model
    train_loss, val_loss = train_and_validate_model(
        model, optimizer, config, train_data, val_data, 
        early_stopper, scheduler, device
    )

    # Log validation metrics and report to raytune
    best_epoch = log_and_report_metrics(val_loss)

    # Load the best model and evaluate
    model = load_best_model(early_stopper, model, device, best_epoch)
    # Train
    evaluate_model_mlflow(model, data, train_data, device, config, name='train')
    # Val 
    evaluate_model_mlflow(model, data, val_data, device, config, name='val')
    # Test
    test_targets, metrics_by_type = evaluate_model_mlflow(model, data, test_data, device, config, name='test')

    # Creates barplot comparison over all metrics for multi target learning
    if test_targets.dim() > 1:
        create_barplot(metrics_by_type, config)

    # Save the final model
    save_final_model(model, config)

    # Log duration of the training process
    log_duration(start_time)

def main():
    set_seed(1)
    # Load config file from its path, in this case its in the same directory as the script
    path = get_path(current_dir, 'config.yaml')
    # Important to set the paths in the config file
    base_config = load_config(path)
    # Define configurations for Ray Tune
    search_space = {
            'epochs': tune.choice([10]), #1000
            'batch_size': tune.choice([None]),#tune.choice([1024,4096,8192]), # None for full dataset
            'hidden_dim': tune.choice([128,256,512,1024]), #tune.choice([47]),
            'dropout': tune.uniform(0.0, 0.1), # tune.uniform(0.0, 0.1),
            'patience': tune.choice([50]), #125
            'patience_scheduler': tune.choice([10]),
            'learning_rate': tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-2), # 0.007902619549708237 tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-3), # tune.choice([0.0008927180304353635]), #tune.loguniform(1e-4, 1e-3), #4e-4 to 5e-4 same here #tune.loguniform(1e-5, 1e-2), #tune.choice([0.001054627]), tune.choice([0.0053708188433723904]) 0.0008927180304353635tune.loguniform(1e-4, 1e-3) tune.choice([0.00023273922280628748])
            'weight_decay': tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-5, 1e-4), # 4e-5 to 5e-5 #tune.loguniform(1e-5, 1e-3),# tune.choice([1e-4]), #tune.choice([0.00045529597892867465]) tune.loguniform(1e-5, 1e-4) tune.choice([0.00006021310185147612])
            'schedule_factor': tune.choice([0.8]), 
            'mlp_layers': tune.choice([2,3,4,5]), #tune.choice([2,3,4,5]), 2 4
            'num_layers_atom': tune.choice([2,3,4,5]), #tune.choice([2,3,4,5]), 3 2
            'num_layers_mol': tune.choice([2,3,4,5]), #tune.choice([2,3,4,5]), 3 3
        }

    config = {**base_config, **search_space}
    
    df, train_data, val_data, test_data, data = load_dataset_from_csv(config, return_dataset=True, limit = 100) # None 100
    # print("train_data.shape", len(train_data))
    # print("train_data[0] ", train_data[0])
    # print("train targets shape", train_data[0].y.shape)
    # breakpoint()
    # Bundle the df & datasets into one variable
    data_bundle = {
        'df': df,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'data': data,
    }
    
    hyperparameter_tuning(run_experiment, config, num_samples=1, storage_path=get_path(current_dir, '../ray_results'), data_bundle=data_bundle)

if __name__ == "__main__":
    main()


# search_space = {
#         'epochs': tune.choice([1000]), #1000
#         'batch_size': tune.choice([None]),#tune.choice([1024,4096,8192]), # None for full dataset
#         'hidden_dim': tune.choice([128,256,512,1024]), #tune.choice([47]),
#         'dropout': tune.uniform(0.0, 0.1), # tune.uniform(0.0, 0.1),
#         'patience': tune.choice([50]), #125
#         'patience_scheduler': tune.choice([10]),
#         'learning_rate': tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-2), # 0.007902619549708237 tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-3), # tune.choice([0.0008927180304353635]), #tune.loguniform(1e-4, 1e-3), #4e-4 to 5e-4 same here #tune.loguniform(1e-5, 1e-2), #tune.choice([0.001054627]), tune.choice([0.0053708188433723904]) 0.0008927180304353635tune.loguniform(1e-4, 1e-3) tune.choice([0.00023273922280628748])
#         'weight_decay': tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-5, 1e-4), # 4e-5 to 5e-5 #tune.loguniform(1e-5, 1e-3),# tune.choice([1e-4]), #tune.choice([0.00045529597892867465]) tune.loguniform(1e-5, 1e-4) tune.choice([0.00006021310185147612])
#         'schedule_factor': tune.choice([0.8]), 
#         'mlp_layers': tune.choice([2,3,4,5]), #tune.choice([2,3,4,5]), 2 4
#         'num_layers_atom': tune.choice([2,3,4,5]), #tune.choice([2,3,4,5]), 3 2
#         'num_layers_mol': tune.choice([2,3,4,5]), #tune.choice([2,3,4,5]), 3 3
#     }


# search_space = {
    #     'epochs': tune.choice([1000]), #1000
    #     'batch_size': tune.choice([None]),#tune.choice([1048,4192,8384]), # None for full dataset
    #     'hidden_dim': tune.choice([128]), #tune.choice([47]),
    #     'dropout': tune.choice([0.006262658491111717]), 
    #     'patience': tune.choice([50]), #125
    #     'patience_scheduler': tune.choice([10]),
    #     'learning_rate': tune.choice([0.0006954802068126907]), #tune.loguniform(1e-4, 1e-3), # tune.choice([0.0008927180304353635]), #tune.loguniform(1e-4, 1e-3), #4e-4 to 5e-4 same here #tune.loguniform(1e-5, 1e-2), #tune.choice([0.001054627]), tune.choice([0.0053708188433723904]) 0.0008927180304353635tune.loguniform(1e-4, 1e-3) tune.choice([0.00023273922280628748])
    #     'weight_decay': tune.choice([2.23492e-6]), #tune.loguniform(1e-5, 1e-4), # 4e-5 to 5e-5 #tune.loguniform(1e-5, 1e-3),# tune.choice([1e-4]), #tune.choice([0.00045529597892867465]) tune.loguniform(1e-5, 1e-4) tune.choice([0.00006021310185147612])
    #     'mlp_layers': tune.choice([3]), #tune.choice([2,3,4,5]), 2 4
    #     'schedule_factor': tune.choice([0.8]), 
    #     'num_layers_atom': tune.choice([3]), #tune.choice([2,3,4,5]), 3 2
    #     'num_layers_mol': tune.choice([5]), #tune.choice([2,3,4,5]), 3 3
    # }

    # search_space = {
    #     'epochs': tune.choice([1000]), #1000
    #     'batch_size': tune.choice([None]),#tune.choice([1024,4096,8192]), # None for full dataset
    #     'hidden_dim': tune.choice([512]), #tune.choice([47]),
    #     'dropout': tune.choice([0.04822474052593628]), #tune.uniform(0.0, 0.1), # tune.uniform(0.0, 0.1), # tune.uniform(0.0, 0.1),
    #     'patience': tune.choice([50]), #125
    #     'patience_scheduler': tune.choice([10]),
    #     'learning_rate': tune.choice([0.002186633528182565]), #tune.loguniform(1e-4, 1e-2), # tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-2), # 0.007902619549708237 tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-2), #tune.loguniform(1e-4, 1e-3), # tune.choice([0.0008927180304353635]), #tune.loguniform(1e-4, 1e-3), #4e-4 to 5e-4 same here #tune.loguniform(1e-5, 1e-2), #tune.choice([0.001054627]), tune.choice([0.0053708188433723904]) 0.0008927180304353635tune.loguniform(1e-4, 1e-3) tune.choice([0.00023273922280628748])
    #     'weight_decay': tune.choice([0.00017831622822312622]), #tune.loguniform(1e-7, 1e-3), # tune.loguniform(1e-7, 1e-3),#tune.loguniform(1e-7, 1e-3), tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-5, 1e-4), # 4e-5 to 5e-5 #tune.loguniform(1e-5, 1e-3),# tune.choice([1e-4]), #tune.choice([0.00045529597892867465]) tune.loguniform(1e-5, 1e-4) tune.choice([0.00006021310185147612])
    #     'schedule_factor': tune.choice([0.8]), 
    #     'mlp_layers': tune.choice([2]), #tune.choice([2,3,4,5]), 2 4
    #     'num_layers_atom': tune.choice([3]), #tune.choice([2,3,4,5]), 3 2
    #     'num_layers_mol': tune.choice([3]), #tune.choice([2,3,4,5]), 3 3
    # }