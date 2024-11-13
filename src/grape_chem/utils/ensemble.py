import torch
from torch.optim import Adam
from torch.nn import MSELoss
from grape_chem.models import AFP, DMPNN
from torch_geometric.data import Data
from grape_chem.utils import train_model, test_model, pred_metric, EarlyStopping
from grape_chem.utils.data import DataSet
from grape_chem.utils.model_utils import set_seed, return_hidden_layers
from grape_chem.plots.ensemble_plots import *
from sklearn.utils import resample
import numpy as np

from matplotlib import pyplot as plt

import mlflow

import seaborn as sns
import os
import json
import pandas as pd

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

def calculate_std_metrics(metrics_dict, technique, labels):
    for idx, label in enumerate(labels):
        metrics = np.array([
            [metric['mse'], metric['rmse'], metric['sse'], metric['mae'], 
            metric['r2'], metric['mre'], metric['mdape']]
            for metric in metrics_dict
        ])
        std_metrics = {
            f'{label}_{technique} mse std': np.std(metrics[idx:, 0]),
            f'{label}_{technique} rmse std': np.std(metrics[idx:, 1]),
            f'{label}_{technique} sse std': np.std(metrics[idx:, 2]),
            f'{label}_{technique} mae std': np.std(metrics[idx:, 3]),
            f'{label}_{technique} r2 std': np.std(metrics[idx:, 4]),
            f'{label}_{technique} mre std': np.std(metrics[idx:, 5]),
            f'{label}_{technique} mdape std': np.std(metrics[idx:, 6])
        }
        mlflow.log_metrics(std_metrics)
    return std_metrics


class Ensemble:
    def __init__(self, train_data, val_data, test_data, node_in_dim, edge_in_dim, device, hyperparams, mean=None, std=None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.device = device
        self.hyperparams = hyperparams
        self.technique = self.__class__.__name__
        self.mean = mean
        self.std = std
        self.UQ_metrics = {
                                'nll': [],
                                'calibrated_nll': [],
                                'mca': [],
                                'spearman_corr': []
                            }

    ## Main ensemble function that runs the ensemble technique and all its helper functions
    def run(self, dataset: DataSet):
        print(f"################ STARTED {self.technique} ################################################")
        train_samples, val_samples = self.create_samples()
        models = []
        for i in range(self.hyperparams['n_models']):
            model = self.train_single_model(train_samples[i], val_samples[i], i)
            models.append(model)
            torch.cuda.empty_cache()
            print("Cleared cache")
        
        self.evaluate_ensemble(models, self.test_data, 'test') 
        print(f"################ FINISHED {self.technique} ################################################")
        torch.cuda.empty_cache()
        print("Cleared cache")
    
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

    def boxplot(self, METRICS, label = 0, metric = 'rmse', dataset_type = 'test'):
        fig, ax = plt.subplots()
        sns.boxplot(data=METRICS, ax=ax)
        ax.set_ylabel(f'{metric}')
        ax.set_xlabel('Model')
        if label != None:
            plot_path = f"{self.technique}_{metric}_{self.hyperparams['data_labels'][label]}_{dataset_type}_boxplot.png"
            plot_title = f"{self.technique} {metric}  boxplot for {self.hyperparams['data_labels'][label]} {dataset_type}"
        else:
            plot_path = f"{self.technique}_{metric}_{dataset_type}_boxplot.png"
            plot_title = f"{self.technique} {metric} boxplot {dataset_type}"
        plt.legend()
        plt.title(plot_title)
        plt.legend()
        plt.savefig(plot_path)
        if mlflow.active_run():
            mlflow.log_artifact(plot_path)

    def get_preds_and_targets_ensemble(self, models, dataset):
        predictions, targets = [], []
        for model in models:
            model.to(self.device)
            pred = test_model(model=model, test_data_loader=dataset, device=self.device, batch_size=len(dataset))
            
            test_targets = [data.y for data in dataset]
            trgts = torch.cat(test_targets, dim=0).to(self.device)

            predictions.append(pred)
            targets.append(trgts)

        return predictions, targets
    
    def get_rescaled_metrics_per_model(self, preds, targets, dataset, num_targets = 1, get_preds = False):
        ''''
        Calculate metrics for each model in the ensemble and optionally return the rescaled predictions and targets.
        '''
        print("Number of targets: ", num_targets)
        metrics = [[] for _ in range(num_targets)]
        rescaled_preds = [[] for _ in range(num_targets)]
        rescaled_targets = [[] for _ in range(num_targets)]
        pred = None
        t = None
        for prediction, target in zip(preds, targets):
            for i in range(num_targets):
                if num_targets > 1:
                    pred = prediction[:, i].cpu().detach()  # Get predictions for the ith target
                    t = target[:, i].cpu().detach()  # Get corresponding true targets
                    mask = ~torch.isnan(t)  # Mask to filter out NaN values
                    pred = pred[mask]
                    t = t[mask]
                    #pred = dataset.rescale_data(pred, index=i)
                    pred = (pred - self.mean[i]) / self.std[i]
                    #t = dataset.rescale_data(t, index=i)
                    t = (t - self.mean[i]) / self.std[i]
                else:
                    pred = prediction.cpu().detach()  # No need to slice for single target
                    t = target.cpu().detach()  # Single target
                    #pred = dataset.rescale_data(pred)
                    pred = (pred - self.mean) / self.std
                    #t = dataset.rescale_data(t)
                    t = (t - self.mean) / self.std
                
                rescaled_preds[i].append(pred)
                rescaled_targets[i].append(t)     
                if torch.isnan(pred).any():
                    print("Predictions contain NaN values.")
                if torch.isnan(t).any():
                    print("Targets contain NaN values.")
                metrics[i].append(pred_metric(pred, t, metrics='all', print_out=False))
                
        if get_preds:
            return rescaled_preds, rescaled_targets, metrics  
        return metrics  
    
    ## Training functions
    def create_model(self):
        # Get MLP hidden layers from configuration
        mlp = return_hidden_layers(self.hyperparams['mlp_layers'])
        if self.hyperparams['model'] == 'afp':
            if self.train_data[0].y.dim() > 1: 
                out_dim = self.train_data[0].y.shape[1]
            elif self.train_data[0].y.dim() == 1:
                out_dim = 1
            return AFP(node_in_dim=self.node_in_dim, edge_in_dim=self.edge_in_dim, out_dim=out_dim, hidden_dim=self.hyperparams['hidden_dim'],
                       mlp_out_hidden=mlp, num_layers_atom=self.hyperparams['num_layers_atom'],
                       num_layers_mol=self.hyperparams['num_layers_mol'], dropout=self.hyperparams['dropout']).to(self.device)   
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

    # We need the dataset to rescale the data
    def evaluate_ensemble(self, models, dataset, dataset_type):
        '''Evaluate the ensemble of models and log the metrics to MLflow.
        Args:
            - models (list): List of trained models
            - dataset (DataSet): The dataset to evaluate the ensemble on
            - dataset_type (str): The type of dataset to evaluate on (train, val, test)
        
        '''
        # added self.test_data for clarity that we are predicting on the test data
        prediction_list, target_list = self.get_preds_and_targets_ensemble(models, self.test_data) 
        # predictions and targets are lists of tensors 
        preds = [prediction.cpu().detach() for prediction in prediction_list]
        targets = [target.cpu().detach() for target in target_list]
        num_targets = targets[0].shape[1] if targets[0].ndim > 1 else 1
        print("list of preds shape: ", len(preds))
        print("Preds shape: ", preds[0].shape)
        print("#################################################")
        print("list of targets shape: ", len(targets))
        print("Targets shape: ", targets[0].shape)
        rescaled_preds, rescaled_targets, metrics = self.get_rescaled_metrics_per_model(preds, targets, dataset, num_targets, get_preds = True)        
        print("############### METRICS HERE ############################")
        print("Metrics: ", metrics[0])
        print("Metrics shape: ", len(metrics[0]))
        print("############### METRICS DONE ############################")
        # Loop over each target index
        for i in range(num_targets):
            # Std of metrics per entire ensemble
            #std_metrics = calculate_std_metrics(metrics[i], self.technique, self.hyperparams['data_labels'])
            print("################################### BOXPLOTS ############################################")
            #print("std metrics: ", std_metrics) 
            # Initialize RMSE array: shape (num_models, num_targets)
            # Initialize empty arrays for metrics
            print("############### METRICS : ", metrics)
            RMSE = np.zeros((len(metrics[0]), num_targets))
            MAE = np.zeros((len(metrics[0]), num_targets))
            R2 = np.zeros((len(metrics[0]), num_targets))

            print("############### METRICS ############################")

            # Retrieve RMSE, MAE, and R2 values for each model for the current target
            for model_idx, model_metrics in enumerate(metrics[i]):
                RMSE[model_idx, 0] = model_metrics['rmse']
                MAE[model_idx, 0] = model_metrics['mae']
                R2[model_idx, 0] = model_metrics['r2']

            if mlflow.active_run():
                # Determine the column index based on the number of targets
                col_idx = 0 if num_targets == 1 else i

                print("RMSE shape: ", RMSE.shape)
                print("MAE shape: ", MAE.shape)
                print("R2 shape: ", R2.shape)

                # Create DataFrame to log metrics for the specified column index
                metrics_df = pd.DataFrame({
                    'RMSE': RMSE[:, col_idx],
                    'MAE': MAE[:, col_idx],
                    'R2': R2[:, col_idx]
                })
                # Save to CSV
                metrics_csv_path =f'{self.technique}_{self.hyperparams["data_labels"][i]}_{dataset_type}_ensemble metrics.csv'
                metrics_df.to_csv(metrics_csv_path, index=False)
                mlflow.log_artifact(metrics_csv_path)

            # metric in max to sum plot needs to be (n,number_targets)
            max_to_sum_plot(self, RMSE, label = i, dataset_type = dataset_type)
            max_to_sum_plot(self, MAE, label = i, dataset_type = dataset_type)
            max_to_sum_plot(self, R2, label = i, dataset_type = dataset_type)

            self.boxplot(RMSE, label = i, metric = 'rmse', dataset_type = dataset_type)
            self.boxplot(MAE, label = i, metric = 'mae', dataset_type = dataset_type)
            self.boxplot(R2, label = i, metric = 'r2' , dataset_type = dataset_type)

            plot_confidence_interval(self, rescaled_preds[i], label=i, dataset_type = dataset_type)
            target = rescaled_targets[i][0]
            # Compute average preds inside error vs uncertainty plot function
            plot_error_vs_uncertainty(self, rescaled_preds[i], np.array(target), label=i, dataset_type = dataset_type)

            rescaled_avg_predictions = torch.mean(torch.stack(rescaled_preds[i]), dim=0)
            metric_for_logging = pred_metric(rescaled_avg_predictions, target, metrics='all', print_out=False)
            # Update the metrics keys
            updated_metrics = {f"{self.hyperparams['data_labels'][i]}_{self.technique}_{key}_{dataset_type}": value for key, value in metric_for_logging.items()}
            mlflow.log_metrics(updated_metrics)
            print("################################### BOXPLOTS done ############################################")
            # # Calc the average predictions
            # rescaled_avg_predictions = torch.mean(torch.stack(rescaled_preds[0]), dim=0)
            # print("Average Predictions: ", rescaled_avg_predictions)
            # self.log_and_plot_ensemble_uq_metrics(rescaled_avg_predictions.cpu().detach(), target.cpu().detach())
        

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
                                          y=synthetic_values[i].unsqueeze(0), revedge_index=getattr(data[i], 'revedge_index', None)) for i in range(len(data))]
                synth_samples.append(synthetic_dataset)
            return synth_samples

        reference_model = self.create_model()
        criterion = MSELoss()
        optimizer = Adam(reference_model.parameters(), lr=self.hyperparams['learning_rate'], weight_decay=self.hyperparams['weight_decay'])
        early_stopping = EarlyStopping(patience=self.hyperparams['early_stopping_patience'])
        train_model(model=reference_model, loss_func=criterion, optimizer=optimizer, train_data_loader=self.train_data,
                    val_data_loader=self.val_data, epochs=self.hyperparams['epochs'], device=self.device, batch_size=self.hyperparams['batch_size'], early_stopper=early_stopping)
        
        train_samples = create_synth_samples(reference_model, self.train_data, self.hyperparams['n_models'])
        val_samples = create_synth_samples(reference_model, self.val_data, self.hyperparams['n_models'])
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