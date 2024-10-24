# grape/hyperparameter_tuning.py
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os

# Shorten directory name to avoid exceeding path length limits
def custom_trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

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
        resources_per_trial={"cpu": 2, "gpu": 1}  # Add this to specify the resources per trial
    )

    print("Best hyperparameters found: ", analysis.best_config)
    return analysis.best_config