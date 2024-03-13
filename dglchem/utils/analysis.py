# analysis tools

import os
import time
import json
import urllib.request
import math

from tqdm import tqdm
import pandas as pd
import rdkit.Chem
from rdkit import Chem
from dgllife.utils.analysis import analyze_mols
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dglchem.utils.feature_func import mol_weight

__all__ = [
    'smiles_analysis',
    'classify_compounds',
    'classyfire',
    'classyfire_result_analysis',
    'num_chart',
    'mol_weight_vs_target',
    'compound_nums_chart',
    'compounds_dataset_heatmap',
    'loss_plot'
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

def classify_compounds(smiles: list) -> tuple:
    """Function that classifies compounds into the following classes:
        ['Hydrocarbons', 'Oxygenated', 'Nitrogenated', 'Chlorinated', 'Fluorinated', 'Brominated', 'Iodinated',
        'Phosphorous containing', 'Sulfonated', 'Silicon containing']

    Author: arnaou

    Parameters
    ----------
    smiles
        List of smiles that will be classified into the families.

    Returns
    -------
    class_dictionary, length_dictionary
        The class dictionary contains the classes and associated indices, the length dictionary contains the
        summarized lengths


    """

    df = pd.DataFrame({'SMILES': smiles})

    # Defining the class names
    class_names = ['Hydrocarbons', 'Oxygenated', 'Nitrogenated', 'Chlorinated', 'Fluorinated', 'Brominated',
                   'Iodinated', 'Phosphorous containing', 'Sulfonated', 'Silicon containing']
    # Defining the class tags
    class_patterns = ['C', 'CO', 'CN', 'CCL', "CF", "CBR", "CI", "CP", "CS", "CSI"]

    class_dict = {}
    for i in class_names:
        class_dict[i] = []
    for j, smi in enumerate(df['SMILES']):
        s = ''.join(filter(str.isalpha, smi)).upper()
        for n in range(len(class_names)):
            allowed_char = set(class_patterns[n])
            if set(s) == allowed_char:
                if class_names[n] == 'Chlorinated':
                    if 'CL' in s:
                        class_dict[class_names[n]].append(j)
                elif class_names[n] == 'Brominated':
                    if 'BR' in s:
                        class_dict[class_names[n]].append(j)
                elif class_names[n] == "Silicon containing":
                    if 'SI' in s:
                        class_dict[class_names[n]].append(j)
                else:
                    class_dict[class_names[n]].append(j)

    sum_lst = []
    for key in class_dict:
        sum_lst.extend(class_dict[key])

        # check the consistence
    if len(sum_lst) == len(list(set(sum_lst))):
        multi_lst = list(set(range(len(df))) - set(sum_lst))
        class_dict['Multifunctional'] = multi_lst
        length_dict = {key: len(value) for key, value in class_dict.items()}
    else:
        raise ValueError('The sum is not matching')

    return class_dict, length_dict

def print_report(string, file=None):
    file.write('\n' + string)

def classyfire(mols: list[rdkit.Chem.Mol], path_to_export: str = None):
    """Applies the classyfire procedure given in [1] to the input molecules and generates json files with the
    information. The procedure will take about ~10min for 100 mols.

    Author: arnaou

    Parameters
    ----------
    mols: list[rdkit.Chem.Mol]
        The input mol objects, their information will be attempted to retrieve from the classyfire database.
    path_to_export: str
        Path where the classyfire results should be stored.

    References
    ----------
    [1] Djoumbou Feunang et al., ClassyFire: automated chemical classification with a comprehensive, computable taxonomy.,
    2016, https://doi.org/10.1186/s13321-016-0174-y

    """

    if path_to_export is None:

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    inchikey_rdkit = []
    for mol in mols:
        try:
            inchikey_rdkit.append(Chem.inchi.MolToInchiKey(mol))
        except:
            inchikey_rdkit.append('')

    # download classification using inchikey
    path_folder = os.path.join(path_to_export,'classyfire')
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    missing_keys = False
    path_report = 'missing_keys.txt'
    report = open(path_report, 'w')

    for i in tqdm(range(len(inchikey_rdkit))):
        key = inchikey_rdkit[i]
        url = 'https://cfb.fiehnlab.ucdavis.edu/entities/' + str(key) + '.json'
        try:
            with urllib.request.urlopen(url) as webpage:
                data = json.loads(webpage.read().decode())

            with open(path_folder + '/' + str(i) + '.json', 'w') as f:
                json.dump(data, f)
        except:
            print_report(str(i) + '    ' + str(key))
            missing_keys = True
            pass

        time.sleep(math.ceil(len(inchikey_rdkit) / 12 / 60))

    report.close()

    if missing_keys:
        print('Some InChikeys were not available. Please check "Missing_ichikeys.txt" file.')
    else:
        os.remove(path_report)


def classyfire_result_analysis(path_to_classyfire: str = None) -> dict:
    """

    Args:
        path_to_classyfire:

    Returns:

    """

    if path_to_classyfire is None:
        path_to_classyfire = os.getcwd() + '/analysis_results' + '/classyfire'

    classes = dict()
    for file in os.listdir(path_to_classyfire):
        file_path = os.path.join(path_to_classyfire, file)
        try:
            class_name = json.load(open(file_path))['class']['name']
            if class_name in classes.keys():
                classes[class_name] += 1
            else:
                classes[class_name] = 1
        except:
            print(f'No class name in the first layer for file: {file}')
            pass

    return classes






def mol_weight_vs_target(smiles: list, target: list, target_name: str = None, fig_height: int = 8,
                         save_fig: bool = False, path_to_export: str =None) -> sns.jointplot:
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

    plot = sns.jointplot(height=fig_height, data=df, x='molecular weight in [g/mol]',y=target_name, color='blue')

    if path_to_export is not None:
        plot.savefig(fname=f'{path_to_export}/molecular_weight_against_{target_name}.svg', format='svg')

    return


def num_chart(num_dict: dict, fig_size: tuple = (14,8), save_fig: bool = False,
                         path_to_export: str = None) -> sns.barplot:

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    x = num_dict.keys()
    y = num_dict.values()

    plt.rcParams.update({'font.size': 10})

    fig, ax = plt.subplots(figsize=fig_size)
    palette = sns.color_palette('muted', n_colors=len(num_dict.keys()))
    ax = sns.barplot(ax=ax, x = x, y= y, hue=x, legend=False, palette=palette)
    ax.tick_params('x', rotation=60)
    ax.set_xlabel('compound class')
    ax.set_ylabel('number of molecules')

    if path_to_export is not None:
        fig.savefig(fname=f'{path_to_export}/compound_distribution.svg', format='svg', bbox_inches='tight')

    return


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

    return

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

def loss_plot(losses, model_names, fig_size: tuple = (10,5),
                              save_fig: bool = False, path_to_export: str = None) -> sns.lineplot:
    """Creates a line plot of different losses on the same scale.

    Parameters
    ----------
    losses: list
        A list containing the losses.
    model_names: list
        List of the dataset names or loss type.
    fig_size: tuple
        The output figure size. Default: (10,10)
    save_fig: bool
        Decides if the plot is saved, is overridden if a path is given. Default: False
    path_to_export: str
        File location to save. Default: None

    Returns
    -------
    sns.lineplot

    """

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    loss_dic = dict()
    for idx, name in enumerate(model_names):
        loss_dic[name] = losses[idx]

    df = pd.DataFrame(loss_dic)

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=df)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    if path_to_export is not None:
        fig.savefig(fname=f'{path_to_export}/loss_plot.svg', format='svg')

    return









