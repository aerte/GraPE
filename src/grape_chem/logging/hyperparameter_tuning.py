# grape/hyperparameter_tuning.py
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os

from ray import train
from ray import tune

import mlflow

# Shorten directory name to avoid exceeding path length limits
def custom_trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

def log_and_report_metrics(val_loss):
    """Log validation metrics and the best epoch to MLflow.
    And report the best validation loss to Ray Tune.
    """
    best_val_loss = min(val_loss)
    best_epoch = val_loss.index(best_val_loss)
    if mlflow.active_run():
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_metric("best_val_loss", best_val_loss)
    metric = {"val_loss": best_val_loss}
    train.report(metric)
    return best_epoch

def hyperparameter_tuning(train_model_experiment, config, num_samples=10, storage_path='../../ray_results', data_bundle=None):
    scheduler = ASHAScheduler(max_t=100, grace_period=10, reduction_factor=2)
    storage_path = 'file:///' + os.path.abspath(storage_path).replace('\\', '/')
    analysis = tune.run(
        tune.with_parameters(train_model_experiment, data_bundle=data_bundle),
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path=storage_path,
        trial_dirname_creator=custom_trial_dirname_creator,  # Add this to shorten the trial directory names
        resources_per_trial={"cpu": config['cpu'], "gpu": config['gpu']}  # Add this to specify the resources per trial
    )

    print("Best hyperparameters found: ", analysis.best_config)
    # Save best hyperparameters to a file
    with open("best_hyperparameters.txt", "w") as f:
        f.write(str(analysis.best_config))
    # save the file to config save path
    with open(os.path.join(config['save_path'], f"best_hyperparameters_{config['experiment_name']}_{config['data_labels']}.txt"), "w") as f:
        f.write(str(analysis.best_config))
    return analysis.best_config