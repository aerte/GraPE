import torch
import datetime
import numpy as np
import os
from typing import Dict
from collections import defaultdict
import matplotlib.pyplot as plt

from grape_chem.utils import train_model, test_model, pred_metric
from grape_chem.utils.data import load_dataset_from_csv, get_path, get_model_dir, save_splits_to_csv
from grape_chem.utils.model_utils import set_seed, create_checkpoint, load_model_from_data, load_model
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

from ray import train
from ray import tune

# Start MLflow server with the following command:
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

current_dir = os.path.dirname(os.path.abspath(__file__))

def track_params(config: Dict, data_bundle: Dict):
    """Log hyperparameters and dataset information to MLflow."""
    if mlflow.active_run():
        mlflow.log_params(config)

        df, train_data, val_data, test_data, data = (
            data_bundle['df'], data_bundle['train_data'], 
            data_bundle['val_data'], data_bundle['test_data'], 
            data_bundle['data']
        )

        mlflow.log_param("Target", config['target'])
        mlflow.log_param("Global Features", config['global_features'])
        mlflow.log_param("Learning Rate", config['learning_rate'])
        mlflow.log_param("smile_example", data.smiles[37])
        mlflow.log_param("atom_example", data[37].x)
        mlflow.log_param("bond_example", data[37].edge_attr)
        mlflow.log_params({
            "dataset_length": len(data),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data)
        })

        if config['save_splits']:
            print("Saving dataset splits now...")
            mlflow.log_artifact(save_splits_to_csv(df, train_data, val_data, test_data, config['save_path']))

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

def log_metrics(val_loss, model, early_stopper):
    """Log validation metrics and the best epoch to MLflow."""
    best_val_loss = min(val_loss)
    best_epoch = val_loss.index(best_val_loss)
    mlflow.log_param("best_epoch", best_epoch)
    mlflow.log_metric("best_val_loss", best_val_loss)
    metric = {"val_loss": best_val_loss}
    train.report(metric)
    return best_epoch

def load_best_model(early_stopper, model, device, best_epoch):
    """Load the best model saved by early stopping."""
    if early_stopper.best_score is not np.inf and early_stopper.save_path and early_stopper.model_name:
        print("Loading the best model from the early stopper. Best epoch: ", best_epoch)
        model_filename = os.path.join(early_stopper.save_path, early_stopper.model_name)
        if os.path.exists(model_filename):
            print(f"Model file '{model_filename}' found. Loading the trained model.")
            model.load_state_dict(torch.load(model_filename, map_location=device).get('model_state_dict'))
            model.eval()
        else:
            print(f"Model file '{model_filename}' not found. Using the last model.")
    else:
        print("No model was saved by the early stopper. Using the last model.")

def evaluate_model(model, data, test_data, device, batch_size, config):
    """Evaluate the model on the test set and log metrics to MLflow."""
    pred = test_model(model=model, test_data_loader=test_data, device=device, batch_size=batch_size, mlflow_log=True)
    targets = test_data.y

    target_tensor = torch.tensor(targets, dtype=torch.float32)
    mean_tensor = torch.tensor(data.mean, dtype=torch.float32).to(pred.device)
    std_tensor = torch.tensor(data.std, dtype=torch.float32).to(pred.device)

    num_targets = target_tensor.shape[1] if target_tensor.dim() > 1 else 1
    metrics_by_type = defaultdict(list)

    for i in range(num_targets):
        preds = pred[:, i].cpu().detach() if num_targets > 1 else pred.cpu().detach()
        target = target_tensor[:, i].cpu().detach() if num_targets > 1 else target_tensor.cpu().detach()
        mean = mean_tensor[i].cpu().detach() if num_targets > 1 else mean_tensor.cpu().detach()
        std = std_tensor[i].cpu().detach() if num_targets > 1 else std_tensor.cpu().detach()

        if num_targets > 1:
            mask = ~torch.isnan(target)
            preds = preds[mask]
            target = target[mask]

        pred_rescaled = (preds * std) + mean
        target_rescaled = (target * std) + mean

        if torch.isnan(pred_rescaled).any() or torch.isnan(target_rescaled).any():
            print("Predictions or targets contain NaN values.")

        metrics = pred_metric(pred_rescaled, target_rescaled, metrics='all')
        for key, value in metrics.items():
            mlflow.log_metric(f"{config['data_labels'][i]}_{key}", value) if num_targets > 1 else mlflow.log_metric(key, value)

        parity_plot_name = f"test_predictions_vs_targets_{config['data_labels'][i]}" if num_targets > 1 else "test_predictions_vs_targets"
        plot_path = parity_plot_mlflow(parity_plot_name, target_rescaled, pred_rescaled.cpu().detach(), config['save_path'])
        mlflow.log_artifact(plot_path)

        for metric, value in metrics.items():
            metrics_by_type[metric].append(value)

    return target_tensor, metrics_by_type

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

def train_model_experiment(config: Dict, data_bundle: Dict):
    """Main function for training the model."""
    with setup_mlflow(config):
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
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Set up training environment
        early_stopper, scheduler = setup_training_environment(config, optimizer)

        # Train and validate the model
        train_loss, val_loss = train_and_validate_model(
            model, optimizer, config, train_data, val_data, 
            early_stopper, scheduler, device
        )

        # Log validation metrics and report to raytune
        best_epoch = log_metrics(val_loss, model, early_stopper)

        # Load the best model and evaluate
        load_best_model(early_stopper, model, device, best_epoch)
        targets, metrics_by_type = evaluate_model(model, data, test_data, device, config['batch_size'], config)

        # Plot and log metrics
        if targets.dim() > 1:
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
    search_space = {
        'epochs': tune.choice([10]), #1000
        'batch_size': tune.choice([None]),#tune.choice([1048,4192,8384]), # None for full dataset
        'hidden_dim': tune.choice([32,128,256]), #tune.choice([47]),
        'dropout': tune.uniform(0.0, 0.2),
        'patience': tune.choice([50]), #125
        'patience_scheduler': tune.choice([10]),
        'learning_rate': tune.loguniform(1e-4, 1e-3), #tune.loguniform(1e-4, 1e-3), # tune.choice([0.0008927180304353635]), #tune.loguniform(1e-4, 1e-3), #4e-4 to 5e-4 same here #tune.loguniform(1e-5, 1e-2), #tune.choice([0.001054627]), tune.choice([0.0053708188433723904]) 0.0008927180304353635tune.loguniform(1e-4, 1e-3) tune.choice([0.00023273922280628748])
        'weight_decay': tune.loguniform(1e-7, 1e-3), #tune.loguniform(1e-5, 1e-4), # 4e-5 to 5e-5 #tune.loguniform(1e-5, 1e-3),# tune.choice([1e-4]), #tune.choice([0.00045529597892867465]) tune.loguniform(1e-5, 1e-4) tune.choice([0.00006021310185147612])
        'mlp_layers': tune.choice([2]), #tune.choice([2,3,4,5]), 2 4
        'schedule_factor': tune.choice([0.8]), 
        'num_layers_atom': tune.choice([5]), #tune.choice([2,3,4,5]), 3 2
        'num_layers_mol': tune.choice([5]), #tune.choice([2,3,4,5]), 3 3
    }
    config = {**base_config, **search_space}
    
    df, train_data, val_data, test_data, data = load_dataset_from_csv(config, return_dataset=True, limit = None) # None 100
    # Bundle the df & datasets into one variable
    data_bundle = {
        'df': df,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'data': data,
    }

    hyperparameter_tuning(train_model_experiment, config, num_samples=1, storage_path=get_path(current_dir, '../ray_results'), data_bundle=data_bundle)

if __name__ == "__main__":
    main()
