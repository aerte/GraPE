# analyis tools

import os
from rdkit import Chem
from dgllife.utils.analysis import analyze_mols
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    'smiles_analysis'
]

def smiles_analysis(smiles, path_to_export=None, download=False, plots = None, save_plots = False, fig_size = None,
                    output_filter=True):
    """ Function to analyze a list of SMILES. Builds on top of:
    https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/utils/analysis.py

    Args:
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

    Returns:
        dictionary
            Summary of the results.
        figures (optional)
            Bar plots of the specified results.

    """

    if download or (path_to_export is None):

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





