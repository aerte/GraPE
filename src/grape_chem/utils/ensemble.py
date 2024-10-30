import torch
from torch.optim import Adam
from torch.nn import MSELoss
from grape_chem.models import AFP, DMPNN
from torch_geometric.data import Data
from grape_chem.utils import train_model, test_model, pred_metric, EarlyStopping
from grape_chem.utils.data import DataSet
from grape_chem.utils.model_utils import set_seed
from grape_chem.plots.ensemble_plots import calculate_nll, calculate_calibrated_nll, calculate_mca, calculate_spearman_rank
from sklearn.utils import resample
import numpy as np

from matplotlib import pyplot as plt

import mlflow

import seaborn as sns
import os

def print_averages(metrics, technique):
    avg_metrics = {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}
    print(f"""Average Metrics for Ensemble Models using {technique}:
    MSE: {avg_metrics['mse']:.6f}
    RMSE: {avg_metrics['rmse']:.6f}
    SSE: {avg_metrics['sse']:.6f}
    MAE: {avg_metrics['mae']:.6f}
    R2: {avg_metrics['r2']:.6f}
    MRE: {avg_metrics['mre']:.6f}
    MDAPE: {avg_metrics['mdape']:.6f}
    """)

def calculate_std_metrics(metrics_dict, technique):
    metrics = np.array([
        [metric['mse'], metric['rmse'], metric['sse'], metric['mae'], 
         metric['r2'], metric['mre'], metric['mdape']]
        for metric in metrics_dict
    ])
    std_metrics = {
        f'{technique} mse std': np.std(metrics[:, 0]),
        f'{technique} rmse std': np.std(metrics[:, 1]),
        f'{technique} sse std': np.std(metrics[:, 2]),
        f'{technique} mae std': np.std(metrics[:, 3]),
        f'{technique} r2 std': np.std(metrics[:, 4]),
        f'{technique} mre std': np.std(metrics[:, 5]),
        f'{technique} mdape std': np.std(metrics[:, 6])
    }
    mlflow.log_metrics(std_metrics)


class Ensemble:
    def __init__(self, train_data, val_data, test_data, node_in_dim, edge_in_dim, device, hyperparams):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.device = device
        self.hyperparams = hyperparams
        self.technique = self.__class__.__name__
        self.UQ_metrics = {
                                'nll': [],
                                'calibrated_nll': [],
                                'mca': [],
                                'spearman_corr': []
                            }

    
    def log_and_plot_ensemble_uq_metrics(self, predictions, targets):
        """Log and plot UQ metrics of the entire ensemble to MLflow."""
        # Plot NLL and Calibrated NLL distribution
        calculate_nll(self, predictions, targets)
        calculate_calibrated_nll(self, predictions, targets)
        calculate_mca(self, predictions, targets)
        calculate_spearman_rank(self, predictions, targets)

        metrics = pred_metric(predictions, targets, metrics='all', print_out=False)
        # Update the metrics keys
        updated_metrics = {f'{self.technique}_{key}': value for key, value in metrics.items()}
        mlflow.log_metrics(updated_metrics)

    def get_preds_and_targets_ensemble(self, models):
        predictions, targets = [], []
        for model in models:
            model.to(self.device)
            pred = test_model(model=model, test_data_loader=self.test_data, device=self.device, batch_size=len(self.test_data))
            
            test_targets = [data.y for data in self.test_data]
            trgts = torch.cat(test_targets, dim=0).to(self.device)

            predictions.append(pred)
            targets.append(trgts)

        return predictions, targets
    
    ## Training functions
    def create_model(self):
        if self.hyperparams['model'] == 'afp':
            if self.train_data[0].y.dim() > 1: 
                out_dim = self.train_data[0].y.shape[1]
            elif self.train_data[0].y.dim() == 1:
                out_dim = 1
            return AFP(node_in_dim=self.node_in_dim, edge_in_dim=self.edge_in_dim, out_dim=out_dim).to(self.device)
        elif self.hyperparams['model'] == 'dmpnn':
            return DMPNN(node_in_dim=self.node_in_dim, edge_in_dim=self.edge_in_dim).to(self.device)
        else:
            raise ValueError('Invalid model type')

    def train_single_model(self, train_data, val_data, model_index):
        print(f'Training model {model_index + 1}/{self.hyperparams["n_models"]}')
        model = self.create_model()
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=self.hyperparams['learning_rate'])
        early_stopping = EarlyStopping(patience=self.hyperparams['early_stopping_patience'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.hyperparams['scheduler_factor'], patience=self.hyperparams['scheduler_patience']
        )

        train_model(model=model, loss_func=criterion, optimizer=optimizer, scheduler=scheduler,
                    train_data_loader=train_data, val_data_loader=val_data, epochs=self.hyperparams['epochs'], device=self.device,
                    batch_size=self.hyperparams['batch_size'], early_stopper=early_stopping)
        model.load_state_dict(torch.load('best_model.pt'))
        return model

    ## Main ensemble function that runs the ensemble technique and all its helper functions
    def run(self, dataset: DataSet):
        train_samples, val_samples = self.create_samples()
        models = []
        for i in range(self.hyperparams['n_models']):
            model = self.train_single_model(train_samples[i], val_samples[i], i)
            models.append(model)
        
        prediction_list, target_list = self.get_preds_and_targets_ensemble(models) 
        
        # Rescale predictions and targets using list comprehensions
        preds = [prediction.cpu().detach() for prediction in prediction_list]
        targets = [target.cpu().detach() for target in target_list]
        metrics = []
        num_targets = targets[0].shape[1] if targets[0].ndim > 1 else 1
        for prediction, target in zip(preds, targets):
            for i in range(num_targets):
                if num_targets > 1:
                    pred = prediction[:, i].cpu().detach()  # Get predictions for the ith target
                    t = target[:, i].cpu().detach()  # Get corresponding true targets
                else:
                    pred = prediction.cpu().detach()  # No need to slice for single target
                    t = target.cpu().detach()  # Single target

                if num_targets > 1:
                    mask = ~torch.isnan(t)  # Mask to filter out NaN values
                    pred = pred[mask]
                    t = t[mask]
                
                pred = dataset.rescale_data(pred)
                t = dataset.rescale_data(t)

                if torch.isnan(pred).any():
                   print("Predictions contain NaN values.")
                if torch.isnan(t).any():
                   print("Targets contain NaN values.")
                
                metrics.append(pred_metric(pred, t, metrics='all', print_out=False))
        calculate_std_metrics(metrics, self.technique)
        
        # Take one sample of the targets
        target = targets[0]
        # Calc the average predictions
        avg_predictions = torch.mean(torch.stack(preds), dim=0)
        if num_targets == 1:
            self.log_and_plot_ensemble_uq_metrics(avg_predictions.cpu().detach(), target.cpu().detach())
        else:
            print("No working plots for multiple targets")

    def create_samples(self):
        raise NotImplementedError("Subclasses should implement this method")

class Bagging(Ensemble):
    def create_samples(self):
        def create_bootstrap_samples(data, n_samples):
            samples = []
            for _ in range(n_samples):
                indices = resample(range(len(data)), replace=True)
                sample_data = [data[i] for i in indices]
                samples.append(sample_data)
            return samples

        train_samples = create_bootstrap_samples(self.train_data, self.hyperparams['n_models'])
        val_samples = create_bootstrap_samples(self.val_data, self.hyperparams['n_models'])
        return train_samples, val_samples

class RandomWeightInitialization(Ensemble):
    import os
    # Set the CUBLAS_WORKSPACE_CONFIG environment variable to ensure deterministic behavior when using GPU operations.
    # This setting is necessary because certain operations in CuBLAS (CUDA's GPU-accelerated library) are non-deterministic by default,
    # which can cause results to vary between runs even with the same inputs.
    # Setting this value (either ":4096:8" or ":16:8") configures the workspace size to guarantee reproducibility, avoiding non-deterministic behavior.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    def create_samples(self):
        train_samples = [self.train_data] * self.hyperparams['n_models']
        val_samples = [self.val_data] * self.hyperparams['n_models']
        return train_samples, val_samples

    def train_single_model(self, train_data, val_data, model_index):
        set_seed(self.hyperparams['weight_seed'] + model_index)
        return super().train_single_model(train_data, val_data, model_index)

class Jackknife(Ensemble):
    def create_samples(self):
        def create_synth_samples(reference_model, data, n_samples):
            synth_samples = []
            reference_model = reference_model.to(self.device)
            with torch.no_grad():
                predictions = test_model(reference_model, test_data_loader=data, device=self.device, batch_size=len(data))
            actual_values = [data[i].y.to(self.device) for i in range(len(data))]
            residuals = [actual_values[i] - predictions[i] for i in range(len(data))]

            for _ in range(n_samples):
                sampled_residuals = torch.stack(resample(residuals, replace=True))
                synthetic_values = predictions + sampled_residuals[0]
                synthetic_dataset = [Data(x=data[i].x, edge_index=data[i].edge_index, edge_attr=data[i].edge_attr,
                                          y=synthetic_values[i], revedge_index=getattr(data[i], 'revedge_index', None)) for i in range(len(data))]
                synth_samples.append(synthetic_dataset)
            return synth_samples

        reference_model = self.create_model()
        criterion = MSELoss()
        optimizer = Adam(reference_model.parameters(), lr=self.hyperparams['learning_rate'])
        early_stopping = EarlyStopping(patience=self.hyperparams['early_stopping_patience'])
        train_model(model=reference_model, loss_func=criterion, optimizer=optimizer, train_data_loader=self.train_data,
                    val_data_loader=self.val_data, epochs=self.hyperparams['epochs'], device=self.device, batch_size=self.hyperparams['batch_size'], early_stopper=early_stopping)
        
        train_samples = create_synth_samples(reference_model, self.train_data, self.hyperparams['n_models'])
        val_samples = [self.val_data] * self.hyperparams['n_models']
        return train_samples, val_samples

class BayesianBootstrap(Ensemble):
    def create_samples(self):
        def Dirichlet_sample(m, n):
            Dirichlet_params = np.ones(m * n)
            Dirichlet_weights = np.asarray([np.random.gamma(a, 1) for a in Dirichlet_params])
            Dirichlet_weights = Dirichlet_weights.reshape(m, n)
            row_sums = Dirichlet_weights.sum(axis=1)
            Dirichlet_weights /= row_sums[:, np.newaxis]
            return Dirichlet_weights

        def create_bayesian_samples(data, n_samples):
            weights_matrix = Dirichlet_sample(n_samples, len(data))
            samples = []
            
            for weights in weights_matrix:
                indices = np.random.choice(len(data), len(data), replace=True, p=weights)
                sampled_data = [data[i] for i in indices]
                samples.append(sampled_data)
            
            return samples

        train_samples = create_bayesian_samples(self.train_data, self.hyperparams['n_models'])
        val_samples = create_bayesian_samples(self.val_data, self.hyperparams['n_models'])
        return train_samples, val_samples