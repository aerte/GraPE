import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from typing import Dict
from scipy.stats import spearmanr

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_error_vs_uncertainty(ensemble, predictions, targets, label = None, dataset_type = 'test'):
    """
    Plots the relationship between prediction error and uncertainty (measured as standard deviation).

    Args:
        - predictions (list of numpy.ndarray): A list of length n, where each element is a numpy array of length m containing predictions from a model.
        - targets (numpy.ndarray): A numpy array of length m containing the true values.
        - label: int, optional, default=None, label of the data from config.
    Returns:
        None: The function generates a plot illustrating the relationship between prediction error and uncertainty.
    """
    # Convert predictions list to a numpy array of shape (n, m)
    predictions_array = np.array(predictions)
    
    # Calculate mean predictions and standard deviation (uncertainty)
    mean_predictions = np.mean(predictions_array, axis=0)
    std_predictions = np.std(predictions_array, axis=0)
    print("################## Predictions ####################")
    print(mean_predictions)
    print("################## Targets ####################")
    print(targets)
    print("################## Errors ####################")
    # Calculate raw prediction errors
    errors = mean_predictions - targets
    print(errors)
    print("######################################")
    # Calculate the percentage of points within one and two standard deviations
    within_one_std = np.sum(np.abs(errors) <= std_predictions) / len(errors) * 100
    within_two_std = np.sum(np.abs(errors) <= 2 * std_predictions) / len(errors) * 100
    
    # Plot the relationship between prediction error and uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(std_predictions, errors, color='black', label='Data Points')
    
    # Plot one and two standard deviation regions
    x = np.linspace(0, np.max(std_predictions), 100)
    plt.fill_between(x, -x, x, color='gray', alpha=0.5, label='1 Std Dev')
    plt.fill_between(x, -2 * x, 2 * x, color='blue', alpha=0.2, label='2 Std Dev')
    
    # Annotate the percentage of points within one and two standard deviations
    plt.text(0.05, 0.95, f'{within_one_std:.2f}% within 1 Std Dev', transform=plt.gca().transAxes, color='black')
    plt.text(0.05, 0.90, f'{within_two_std:.2f}% within 2 Std Dev', transform=plt.gca().transAxes, color='blue')
    
    # Plot settings
    plt.xlabel('Uncertainty (Standard Deviation)')
    plt.ylabel('Prediction Error')

    if label != None:
        plot_path = f"{ensemble.technique}_{ensemble.hyperparams['data_labels'][label]}_error_vs_uncertainty_{dataset_type}.png"
        plot_title = f"{ensemble.technique} error vs uncertainty for {ensemble.hyperparams['data_labels'][label]}_{dataset_type}"
    else:
        plot_path = f"{ensemble.technique}_error_vs_uncertainty_{dataset_type}.png"
        plot_title = f"{ensemble.technique} error vs uncertainty {dataset_type}"
    plt.title('')
    plt.legend()
    plt.title(plot_title)
    plt.legend()
    plt.savefig(plot_path)
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)


def max_to_sum_plot(ensemble, metrics, p=4, label = None, metric = 'RMSE', dataset_type = 'test'):
    """
    Generates a Max to Sum (M/S) plot for different moments.

    Args:
        - Ensemble
        - metrics (numpy.ndarray): A 2D numpy array of shape (a, b) containing the metrics (e.g., RMSE).
        - p (int): The number of moments to consider (default is 4).

    Returns:
        - None: The function generates a plot illustrating the Max to Sum (M/S) relationship for different moments.
    """
    print("Metrics shape: ", metrics.shape)
    print("Metrics: ", metrics)
    print("##############################################################")
    a, b = metrics.shape  # Get the shape of the input array
    plt.figure(figsize=(10, 6))
    a1 = []  # Initialize list to store lengths

    for j in range(b):
        # Find indices where y is positive
        id = np.where(metrics[:, j] > 0)[0]
        a1.append(len(id))
        # Select one output
        d = metrics[id, j]
        R = []
        for i in range(1, p+1):  # Range from 1 to p
            x = d ** i
            S = np.cumsum(x)
            M = np.maximum.accumulate(x)  # equivalent to cummax
            R.append(M/S)
        R = np.column_stack(R)  # Convert list of arrays to 2D array

        # Plot each moment
        for i in range(p):
            plt.plot(R[:, i], label=f'Moment {i+1} (Output {j+1})')

    plt.xlabel('Index')
    plt.ylabel('Max to Sum Ratio')
    if label != None:
        plot_path = f"{ensemble.technique}_{ensemble.hyperparams['data_labels'][label]}_{dataset_type}_Moment_Analysis.png"
        plot_title = f"{ensemble.technique} _Moment_Analysis {metric} for {ensemble.hyperparams['data_labels'][label]} {metric} {dataset_type}"
    else:
        plot_path = f"{ensemble.technique}_{dataset_type}_Moment_Analysis.png"
        plot_title = f"{ensemble.technique}_Moment_Analysis {metric} {dataset_type}"
    plt.title('')
    plt.legend()
    plt.title(plot_title)
    plt.legend()
    plt.savefig(plot_path)
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)

## ENSEMBLE Logging functions
def calculate_nll(ensemble, predictions, targets):
    """Calculate the Negative Log-Likelihood (NLL) of the predictions."""
    nll = torch.nn.functional.mse_loss(predictions, targets)
    ensemble.UQ_metrics['nll'].append(nll)
    # plot the distributions_
    plt.figure()
    sns.kdeplot(predictions, color="r", label="Predictions")
    sns.kdeplot(targets, color="b", label="Targets")
    plt.title('Average Predictions Ensemble vs Targets Distribution, NLL: {:.4f}'.format(nll.item()))
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plot_path = f"{ensemble.technique}_predictions_targets_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)
    os.remove(plot_path)

def calculate_calibrated_nll(ensemble, predictions, targets):
    """Calculate the Calibrated NLL of the predictions."""
    scaling_factor = torch.std(targets) / torch.std(predictions)
    calibrated_predictions = predictions * scaling_factor
    calibrated_nll = torch.nn.functional.mse_loss(calibrated_predictions, targets)
    ensemble.UQ_metrics['calibrated_nll'].append(calibrated_nll)

    # plot the distributions
    plt.figure()
    sns.kdeplot(calibrated_predictions, color="r", label="Calibrated Predictions")
    sns.kdeplot(targets, color="b", label="Targets")
    plt.title('Calibrated Average Predictions Ensemble vs Targets Distribution, Calibrated NLL: {:.4f}'.format(calibrated_nll.item()))
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plot_path = f"{ensemble.technique}_calibrated_predictions_targets_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)
    os.remove(plot_path)

def calculate_mca(ensemble, predictions, targets):
    """Calculate the Miscalibration Area (MCA) of the predictions."""
    # A placeholder implementation of MCA, which should be based on calibration plot areas
    calibration_error = torch.abs(predictions.mean() - targets.mean())
    ensemble.UQ_metrics['mca'].append(calibration_error)

    plt.figure()
    sns.scatterplot(x=targets, y=predictions, color="b", alpha=0.6, edgecolor=None)
    plt.plot([targets.min(), targets.max()], [predictions.min(), predictions.max()], color='r', linestyle='--', label='Perfect Calibration')
    plt.title(f'Miscalibration Area {ensemble.technique}: {calibration_error.item()}')
    plt.xlabel('Targets')
    plt.ylabel('Average Predictions Ensemble')
    plt.legend()
    plot_path = f"{ensemble.technique}_mca.png"
    plt.savefig(plot_path)
    plt.close()
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)
    os.remove(plot_path)

def calculate_spearman_rank(ensemble, predictions, targets):
    """Calculate Spearman's Rank Correlation Coefficient."""
    targets_np = targets.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    spearman_corr, _ = spearmanr(predictions_np, targets_np)
    ensemble.UQ_metrics['spearman_corr'].append(spearman_corr)

    plt.figure()
    sns.scatterplot(x=targets_np, y=predictions_np, color="b", alpha=0.6, edgecolor=None)
    sns.regplot(x=targets_np, y=predictions_np, color='r', scatter=False, label=f'Spearman Correlation: {spearman_corr:.4f}')
    plt.title(f'Spearman Correlation {ensemble.technique}: {spearman_corr:.4f}')
    plt.xlabel('Targets')
    plt.ylabel('Average Predictions Ensemble')
    plt.legend()
    plot_path = f"{ensemble.technique}_spearman_corr.png"
    plt.savefig(plot_path)
    plt.close()
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)
    os.remove(plot_path)

def plot_confidence_interval(ensemble, predictions, quantiles = 0.95, label = None, dataset_type = 'test'):
    """
    Plots sorted mean predictions with default 95% confidence intervals.
    
    Parameters:
    - predictions: np.ndarray of shape (n_samples, n_models), ensemble predictions for each sample.
    - quantiles: float, optional, default=None, quantile for the confidence interval.
    - label: int, optional, default=None, label of the data from config.
    """
    # Convert predictions to numpy array
    #print("Predictions: ", predictions)
    predictions = np.array(predictions)
    #for pred in predictions:
    #    print("Shape of prediction inside predictions: ", pred.shape)
    #    print("type:", type(pred))
    #print("predictions shape: ", predictions.shape)
    #print("predictions type: ", type(predictions))
    # Calculate mean prediction
    mean_prediction = np.mean(predictions, axis=0)
    lower_bound = (1 - quantiles)/2
    upper_bound = 1 - lower_bound

    # Calculate lower and upper percentiles for confidence interval
    lower_quantile = np.quantile(predictions, lower_bound, axis=0)
    upper_quantile = np.quantile(predictions, upper_bound, axis=0)

    # Sort indices based on mean prediction
    sorted_indices = np.argsort(mean_prediction)

    # Sort mean prediction and quantiles
    mean_prediction_sorted = mean_prediction[sorted_indices]
    lower_quantile_sorted = lower_quantile[sorted_indices]
    upper_quantile_sorted = upper_quantile[sorted_indices]

    n_observations = predictions.shape[1]
    #print("Mean Prediction Sorted: ", mean_prediction_sorted)

    plt.figure(figsize=(10, 6))
    # dots in stead of line
    plt.scatter(range(n_observations), mean_prediction_sorted, label='Mean Prediction', color='blue')
    plt.fill_between(range(n_observations), lower_quantile_sorted, upper_quantile_sorted, color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Observation Index')
    plt.ylabel('Prediction')
    
    if label != None:
        plot_path = f"{ensemble.technique}_{ensemble.hyperparams['data_labels'][label]}_{dataset_type}_CI_quantiles.png"
        plot_title = f"{ensemble.technique} with Confidence Interval for {ensemble.hyperparams['data_labels'][label]} {dataset_type}"
    else:
        plot_path = f"{ensemble.technique}_{dataset_type}_CI_quantiles.png"
        plot_title = f"{ensemble.technique} Ensemble Predictions with Confidence Interval {dataset_type}"
    plt.title(plot_title)
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)