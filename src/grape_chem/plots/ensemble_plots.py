import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from typing import Dict
from scipy.stats import spearmanr

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