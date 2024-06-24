# analysis tools

import os

import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from grape_chem.utils.feature_func import mol_weight
from grape_chem.analysis import classify_compounds
from grape_chem.utils.data import DataSet
from grape_chem.analysis import num_heavy_atoms

__all__ = ['mol_weight_vs_target',
           'num_chart',
           'compound_nums_chart',
           'compounds_dataset_heatmap',
           'num_heavy_plot']

def mol_weight_vs_target(smiles: list, target: list, target_name: str = None, fig_height: int = 8,
                         save_fig: bool = False, path_to_export: str =None,
                         rescale_data: DataSet = None, **kwargs) -> sns.jointplot:
    """Plots a seaborn jointplot of the target distribution against the molecular weight distributions.

    Parameters
    ----------
    smiles: list of str
        SMILES list.
    target: list of float
        Prediction target.
    target_name: str
        Title of the y-axis in the plot.
    fig_height: int
        Determines the figure size of the plot.
    save_fig: bool
        Decides if the figure is saved as a svg (unless a path is given, then it will save the image). Default: False
    path_to_export: str
        File path of the saved image. Default: None
    **kwargs
        Any additional keyword arguments are passed to the sns.jointplot function.


    Returns
    -------
    plot
        The seaborn jointplot object containing the plot.

    """

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    target_name = 'target' if target_name is None else target_name
    weight = np.zeros(len(smiles))

    if rescale_data is not None:
        target = rescale_data.rescale_data(target)

    for i in range(len(smiles)):
        weight[i] = mol_weight(Chem.MolFromSmiles(smiles[i]))

    df = pd.DataFrame({'molecular weight in [g/mol]': weight, target_name: target})

    plot = sns.jointplot(height=fig_height, data=df, x='molecular weight in [g/mol]',y=target_name, color='blue',
                         **kwargs)

    if path_to_export is not None:
        plot.savefig(fname=f'{path_to_export}/molecular_weight_against_{target_name}.svg', format='svg')

    return


def num_chart(num_dict: dict, fig_size: tuple = (14,8), fontsize:int = 10, save_fig: bool = False,
                         path_to_export: str = None) -> sns.barplot:
    """

    Parameters
    ----------
    num_dict: dict
        Dictionary of whatever should be plotted. The keys can be anything, but the values have to be
        integers.
    fig_size: tuple
        The output figure size. Default: (14,8)
    fontsize: int
        The font size. Default: 10
    save_fig: bool
        Decides if the plot is saved, is overridden if a path is given. Default: False
    path_to_export: str
        File location to save. Default: None

    Returns
    -------

    """

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    x = np.array(list(num_dict.keys()))
    # Sort keys for consistency
    x = np.sort(x)

    y = num_dict.values()

    plt.rcParams.update({'font.size': fontsize})

    fig, ax = plt.subplots(figsize=fig_size)
    palette = sns.color_palette('muted', n_colors=len(num_dict.keys()))
    ax = sns.barplot(ax=ax, x = x, y= y, hue=x, legend=False, palette=palette)
    ax.tick_params('x', rotation=90)
    ax.set_xlabel('compound class')
    ax.set_ylabel('number of molecules')

    if path_to_export is not None:
        fig.savefig(fname=f'{path_to_export}/compound_distribution.svg', format='svg', bbox_inches='tight')

    return fig, ax


def compound_nums_chart(smiles: list, fig_size: tuple = (14,8), save_fig: bool = False, path_to_export: str = None) \
        -> sns.barplot:
    """Creates a barchart visualizing the number of molecules classified into different compound types.

    Parameters
    ----------
    smiles: list
        The SMILES that will be classified into compounds and the plotted using a barchart.
    fig_size: tuple
        The output figure size. Default: (14,8)
    save_fig: bool
        Decides if the plot is saved, is overridden if a path is given. Default: False
    path_to_export: str
        File location to save. Default: None

    Returns
    -------
    sns.barplot

    """
    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    _, num_dict = classify_compounds(smiles)
    x = num_dict.keys()
    y = num_dict.values()

    plt.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots(figsize=fig_size)
    palette = sns.color_palette('muted', n_colors=len(num_dict.keys()))
    ax = sns.barplot(ax=ax, x = x, y= y, hue=x, legend=False, palette=palette)
    ax.tick_params('x', rotation=45)
    ax.set_xlabel('compound class')
    ax.set_ylabel('number of molecules')

    if path_to_export is not None:
        fig.savefig(fname=f'{path_to_export}/compound_distribution.svg', format='svg', bbox_inches='tight')

    return ax

def compounds_dataset_heatmap(dataset_smiles: list, dataset_names: list,  fig_size: tuple = (10,10),
                              save_fig: bool = False, path_to_export: str = None) -> sns.heatmap:
    """Creates a heatmap visualizing the number elements in different compounds classes between multiple datasets.

    Parameters
    ----------
    dataset_smiles: list
        A list containing the smiles of the datasets to be considered. Should have the shape like:
        [[smiles_1],[smiles_2]]
    dataset_names: list
        List of the dataset names.
    fig_size: tuple
        The output figure size. Default: (10,10)
    save_fig: bool
        Decides if the plot is saved, is overridden if a path is given. Default: False
    path_to_export: str
        File location to save. Default: None

    Returns
    -------
    sns.heatmap

    """

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)


    results = []
    for smiles in dataset_smiles:
        results_temp = classify_compounds(smiles)[1]
        for key in results_temp.keys():
            results_temp[key] = round(results_temp[key]/len(smiles),2)*100
        results.append(results_temp)
    df = pd.DataFrame(results)
    df.index = dataset_names

    cmap = sns.cm.rocket
    fig, ax = plt.subplots(figsize=fig_size)
    ax = sns.heatmap(ax = ax, data = df, cmap=cmap)

    #if len(dataset_names) < 4:
    #    ax.set_aspect("equal")

    #ax.tick_params('x', rotation=45)
    ax.set_xlabel('compound classes')
    ax.set_ylabel('datasets')

    if path_to_export is not None:
        fig.savefig(fname=f'{path_to_export}/dataset_heatmap.svg', format='svg', bbox_inches='tight')

    return



def num_heavy_plot(smiles: list,  fig_size: tuple = (10,10), save_fig: bool = False, path_to_export: str = None,
                   fontsize:int = 20) -> plt.axes:
    """A histogram of heavy atoms per molecule. Takes a list of smiles as input.

    Parameters
    ----------
    smiles: list
        A list containing the smiles of the datasets to be considered.
    fig_size: tuple
        The output figure size. Default: (10,10)
    save_fig: bool
        Decides if the plot is saved, is overridden if a path is given. Default: False
    path_to_export: str
        File location to save. Default: None
    fontsize:int
        The font size of the figure. Default: 20


    Returns
    -------
    plt.axes

    """

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)


    class_dict = num_heavy_atoms(smiles)

    fig, ax = num_chart(class_dict, fig_size=fig_size, save_fig=False)

    ax.set_xlabel('Number of heavy atoms per molecule', fontsize=fontsize)
    ax.set_ylabel('Counts', fontsize=fontsize)
    ax.set_title('Heavy atoms distribution', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)  # Set fontsize for major tick labels
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)  # Set fontsize for minor tick labels


    if path_to_export is not None:
        fig.savefig(fname=f'{path_to_export}/num_heavy_atoms.svg', format='svg', bbox_inches='tight')

    return ax

