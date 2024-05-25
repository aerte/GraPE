# analysis tools

import os
import pandas as pd
from rdkit import Chem
from dgllife.utils.analysis import analyze_mols
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    'smiles_analysis',
    'classify_compounds',
    'num_heavy_atoms'
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
    """Function that classifies compounds based on SMILES string into the following classes:
        ['Hydrocarbons', 'Oxygenated', 'Nitrogenated', 'Chlorinated', 'Fluorinated', 'Brominated', 'Iodinated',
        'Phosphorous containing', 'Sulfonated', 'Silicon containing']

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


def num_heavy_atoms(smiles: list) -> dict:
    """Simple function that counts the number of heavy atoms per molecule and outputs a dictionary
    that contains the distributions.

    Parameters
    ----------
    smiles

    Returns
    -------

    """

    class_dict = dict()
    for element in smiles:
        mol = Chem.MolFromSmiles(element)
        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy in class_dict:
            class_dict[num_heavy] += 1
        else:
            class_dict[num_heavy] = 1

    return class_dict

