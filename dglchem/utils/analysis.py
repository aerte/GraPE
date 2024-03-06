# analysis tools

import os

import pandas as pd
from rdkit import Chem
from dgllife.utils.analysis import analyze_mols
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dglchem.utils.feature_func import mol_weight
from dglchem import utils

__all__ = [
    'smiles_analysis',
    'mol_weight_vs_target',
    'compound_nums_chart',
    'compounds_dataset_heatmap'
]

def smiles_analysis(smiles: list, path_to_export: str =None, download: bool =False, plots: list = None,
                    save_plots:bool = False, fig_size: list = None, output_filter: bool =True) -> (dict, plt.figure):
    """Function to analyze a list of SMILES. Builds on top of:
    https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/utils/analysis.py.

    Parameters
    ----------
    smiles: list of str
        List of SMILES that are to be analyzed.
    path_to_export: str
        Path to the folder where analysis results should be saved. Default: None
    download: bool
        Decides if the results are downloaded. If either the path is given or download is set to true, the
        analysis results will be downloaded as a txt file. Default: False
    plots: list of str
        Bar plots of the analysis results. Default: None. Possible options are:
        ['atom_type_frequency', 'degree_frequency', 'total_degree_frequency', 'explicit_valence_frequency',
        'implicit_valence_frequency', 'hybridization_frequency', 'total_num_h_frequency', 'formal_charge_frequency',
        'num_radical_electrons_frequency', 'aromatic_atom_frequency', 'chirality_tag_frequency',
        'bond_type_frequency', 'conjugated_bond_frequency', 'bond_stereo_configuration_frequency',
        'bond_direction_frequency']
    save_plots: bool
        Decides if the plots are saved in the processed folder. Default: False
    fig_size: list
        2-D list to set the figure sizes. Default: [10,6]
    output_filter: bool
        Filters the output of excessive output. Default: True (recommended).

    Returns
    -------
    dictionary
        Summary of the results.
    figures (optional)
        Bar plots of the specified results.

    """

    if download and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    dic = analyze_mols(smiles, path_to_export=path_to_export)

    # Filters some non
    if output_filter:
        for key in ['num_atoms', 'num_bonds','num_rings','num_valid_mols','valid_proportion', 'cano_smi']:
            del dic[key]

    if path_to_export is not None:
        if os.path.exists(path_to_export +'/valid_canonical_smiles.txt'):
            os.remove(path_to_export +'/valid_canonical_smiles.txt')

    if plots is not None:

        figs = []
        n = 100
        cmap = plt.cm.get_cmap('plasma', n)

        if fig_size is None:
            fig_size = [10, 6]

        for key in plots:

            if key not in dic.keys():
                print(f'Error: {key} is not a valid plot.')
                continue

            freq = list(dic[key].keys())
            values = list(dic[key].values())

            color = np.random.randint(0,100)

            fig = plt.figure(figsize=fig_size)

            # creating the bar plot
            plt.bar(freq, values, color=cmap(color),
                    width=0.4)

            plt.xlabel(key)
            plt.ylabel(f"{key} in the dataset")

            plt.title("Dataset analysis")

            if save_plots:
                plt.savefig(path_to_export+f'/{key}.png')

            figs.append(fig)

        return dic, figs


    return dic

def mol_weight_vs_target(smiles: list, target: list, target_name: str = None, save_fig: bool = False,
                         path_to_export: str =None) -> sns.jointplot:
    """Plots a seaborn jointplot of the target distribution against the molecular weight distribution.

    Parameters
    ----------
    smiles: list of str
        SMILES list.
    target: list of float
        Prediction target.
    target_name: str
        Title of the y-axis in the plot.
    save_fig: bool
        Decides if the figure is saved as a svg (unless a path is given, then it will save the image). Default: False
    path_to_export: str
        File path of the saved image. Default: None


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

    for i in range(len(smiles)):
        weight[i] = mol_weight(Chem.MolFromSmiles(smiles[i]))

    df = pd.DataFrame({'molecular weight in [g/mol]': weight, target_name: target})

    plot = sns.jointplot(data=df, x='molecular weight in [g/mol]',y=target_name, color='orange')

    if path_to_export is not None:
        plot.savefig(fname=f'{path_to_export}/molecular_weight_against_{target_name}.svg', format='svg')

    return plot


def compound_nums_chart(smiles: list, fig_size: tuple = (14,8), save_fig: bool = False, path_to_export: str = None) \
        -> sns.barplot:
    """

    Parameters:
    ----------
    smiles: list
        The SMILES that will be classified into compounds and the plotted using a barchart.
    save_fig: bool
        Decides if the plot is saved, is overridden if a path is given. Default: False
    path_to_export: str
        File location to save. Default: None

    """
    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    _, num_dict = utils.classify_compounds(smiles)
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

    return

def compounds_dataset_heatmap(dataset_smiles: list, dataset_names: list,  fig_size: tuple = (10,10),
                              save_fig: bool = False, path_to_export: str = None) -> sns.barplot:
    """

    Parameters
    ----------
    dataset_smiles:
    dataset_names:
    fig_size:
    save_fig:
    path_to_export:

    Returns
    -------

    """

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)


    results = []
    for smiles in dataset_smiles:
        results_temp = utils.classify_compounds(smiles)[1]
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







