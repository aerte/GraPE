# -*- coding: utf-8 -*-
#
# Graph constructor built on top of pytorch geometric

import os
from collections.abc import Sequence

import pickle
import pandas as pd
import numpy as np
import torch
from torch import Tensor

from rdkit import Chem
from rdkit.Chem import rdmolops, MolFromSmiles
from rdkit import RDLogger


from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from dglchem.utils.featurizer import AtomFeaturizer, BondFeaturizer
from dglchem.utils.analysis import smiles_analysis, mol_weight_vs_target
from dglchem.utils.data_splitting import split_data

RDLogger.DisableLog('rdApp.*')

__all__ = ['filter_smiles',
           'construct_dataset',
           'classify_compounds',
           'DataLoad',
           'DataSet',
           'GraphDataSet']

def filter_smiles(smiles: list, target: list, allowed_atoms: list = None, log: bool = False) -> (list,list):
    """Filters a list of smiles based on the allowed atom symbols.

    Args
    ----------
    smiles: list of str
        Smiles to be filtered.
    target: list
        Target of the graphs.
    allowed_atoms: list of str
        Valid atom symbols, non-valid symbols will be discarded. Default: [``C``, ``N``, ``O``, ``S``, ``F``, ``Cl``,
         ``Br``, ``I``, ``P``]
    log: bool
        Determines if there should be print-out statements to indicate why mols were filtered out. Default: False

    Returns
    ----------
    list[str]
        A list of filtered smiles strings.

    """

    if allowed_atoms is None:
        allowed_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']

    df = pd.DataFrame({'smiles': smiles, 'target': target})
    indices_to_drop = []

    for element in smiles:
        mol = Chem.MolFromSmiles(element)

        if mol is None:
            if log:
                print(f'SMILES {element} in index {list(df.smiles).index(element)} is not valid.')
            indices_to_drop.append(list(df.smiles).index(element))

        else:
            if mol.GetNumHeavyAtoms() < 2:
                if log:
                    print(f'SMILES {element} in index {list(df.smiles).index(element)} consists of less than 2 heavy atoms'
                        f' and will be ignored.')
                indices_to_drop.append(list(df.smiles).index(element))

            else:
                for atoms in mol.GetAtoms():
                    if atoms.GetSymbol() not in allowed_atoms:
                        if log:
                            print(f'SMILES {element} in index {list(df.smiles).index(element)} contains the atom {atoms.GetSymbol()} that is not'
                                f' permitted and will be ignored.')
                        indices_to_drop.append(list(df.smiles).index(element))

    df.drop(indices_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return list(df.smiles), list(df.target)

def classify_compounds(smiles: list) -> tuple:
    """Function that classifies compounds into the following classes:
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



def construct_dataset(smiles: list, target: list, allowed_atoms: list = None,
                      atom_feature_list: list = None, bond_feature_list: list = None) -> Data:
    """Constructs a dataset out of the smiles and target lists based on the feature lists provided. The dataset will be
    a list of torch geometric Data objects, using their conventions.

    Parameters
    ----------
    smiles : list of str
        Smiles that are featurized and passed into a PyG DataSet.

    target: Any
        Array of values that serve as the graph 'target'.

    allowed_atoms : list of str
        Smiles that are considered in featurization. Default: [``C``, ``N``, ``O``, ``S``, ``F``, ``Cl``, ``Br``,
        ``I``, ``P``]

    atom_feature_list : list of str
        Features of the featurizer, see utils.featurizer for more details. Default: All implemented features.

    bond_feature_list : list of str
        Bond features of the bond featurizer, see utils.featurizer for more details. Default: All implemented features.

    Returns
    -------
    data
        list of Pytorch-Geometric Data objects

    """

    atom_featurizer = AtomFeaturizer(allowed_atoms=allowed_atoms,
                                                atom_feature_list = atom_feature_list)

    bond_featurizer = BondFeaturizer(bond_feature_list=bond_feature_list)


    data = []

    for (smile, i) in zip(smiles, range(len(smiles))):
        mol = MolFromSmiles(smile)
        edge_index = dense_to_sparse(torch.tensor(rdmolops.GetAdjacencyMatrix(mol)))[0]
        x = atom_featurizer(mol)
        edge_attr = bond_featurizer(mol)
        data.append(Data(x = Tensor(x), edge_index = Tensor(edge_index), edge_attr = Tensor(edge_attr),
                         y=Tensor([target[i]])))

    return data


class DataLoad(object):
    """The basic data-loading class. It holds the root as well as the raw  and processed files directories.

    """
    def __int__(self, root = None):
        if root is None:
            root = './data'
        self.root = root

    @property
    def raw_dir(self):
        """

        Returns: path

        """
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        """

        Returns: path

        """
        return os.path.join(self.root, 'processed')




class DataSet(DataLoad):
    """A class that takes a path to a pickle file or a list of smiles and targets. The data is stored in
        Pytorch-Geometric Data instances and be accessed like an array.

        Heavily inspired by the PyTorch-Geometric Dataset class, especially indices and index_select.

    Parameters
    ----------
    file_path: str
        The path to a pickle file that should be loaded and the data therein used.
    smiles: list of str
        List of smiles to be made into a graph.
    target: list of in or float
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list of str
        List of allowed atom symbols.
    atom_feature_list: list of str
        List of features to be applied. Default: All implemented features.
    bond_feature_list: list of str
        List of features that will be applied. Default: All implemented features.
    log: bool
        Decides if the filtering output and other outputs will be shown.


    """
    def __init__(self, file_path = None, smiles = None, target = None, global_features = None, allowed_atoms = None,
                 atom_feature_list = None, bond_feature_list = None, log = False, root = None, indices = None):

        assert (file_path is not None) or (smiles is not None and target is not None),'path or (smiles and target) must given.'

        super().__int__(root)

        if file_path is not None:
            with open(file_path, 'rb') as handle:
                self.data = pickle.load(handle)
                print('Loaded data.')
        else:
            self.smiles, self.raw_target = filter_smiles(smiles, target, allowed_atoms= allowed_atoms, log=log)

            # standardize target
            target_ = np.array(self.raw_target)
            self.target = (target_-np.mean(target_))/np.std(target_)

            self.data = construct_dataset(smiles=self.smiles,
                                          target=self.target,
                                          allowed_atoms = allowed_atoms,
                                          atom_feature_list = atom_feature_list,
                                          bond_feature_list = bond_feature_list)

            self.global_features = global_features
            self._indices = indices
            self.num_node_features = self.data[0].num_node_features
            self.num_edge_features = self.data[0].num_edge_features



    def save_data_set(self, filename=None):
        """Saves the dataset in the processed folder as a pickle file.

        Parameters
        ----------
        filename: str
            Name of the data file. Default: 'data'


        """
        filename='data' if filename is None else filename

        path = self.processed_dir

        if not os.path.exists(path):
            os.makedirs(path)

        with open(path+'/'+filename+'.pickle', 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'File saved at: {path}/{filename}.pickle')


    def get_smiles(self, path=None):
        """

        Parameters
        ----------
        path: str
            Path where the smiles should be saved. Default: The processed files' directory.

        """

        path = self.processed_dir if path is None else path

        if not os.path.exists(path):
            os.makedirs(path)

        np.savetxt(path+'/filtered_smiles.txt', X = np.array(self.smiles), fmt='%s')

    def indices(self):
        """

        Returns
        -------
        list
            Indices of the dataset

        """

        return range(len(self.data)) if self._indices is None else self._indices

    def __len__(self):
        """

        Returns
        -------
        int
            Length of the dataset.

        """
        return len(self.data)

    def __getitem__(self, idx):
        """

        Parameters
        ----------
        idx: int
            Index of item to be returned.

        Returns
        -------
        obj
            Data object at index [item].

        """

        if isinstance(idx, (int, np.integer)):
            return self.data[idx]
        else:
            return self.index_select(idx)

    def __iter__(self):
        """

        Returns:

        """

        for i in range(len(self.data)):
            yield self.data[i]

    def index_select(self, idx):
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.

        Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset

        Parameters
        ----------
            idx: obj
                Index list of data objects to retrieve.

        Returns
        -------
        list
            Python list of data objects.

        """

        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self[i] for i in indices]


    def analysis(self, path_to_export=None, download=False, plots = None, save_plots = False, fig_size=None,
                 output_filter = True):
        """Returns an overview of different aspects of the smiles dataset **after filtering** according to:
        https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/utils/analysis.py.
        This includes the frequency of symbols, degree frequency and more.

        Parameters
        ----------
        path_to_export: str
            Path to the folder where analysis results should be saved. Default: None (recommended).
        download: bool
            Decides if the results are downloaded. If either the path is given or download is set to true, the
            analysis results will be downloaded as a txt file.
        plots: list of str
            Bar plots of the analysis results. Default: None. Possible options are:
            ['atom_type_frequency', 'degree_frequency', 'total_degree_frequency', 'explicit_valence_frequency',
            'implicit_valence_frequency', 'hybridization_frequency', 'total_num_h_frequency', 'formal_charge_frequency',
            'num_radical_electrons_frequency', 'aromatic_atom_frequency', 'chirality_tag_frequency',
            'bond_type_frequency', 'conjugated_bond_frequency', 'bond_stereo_configuration_frequency',
            'bond_direction_frequency']
        save_plots: bool
            Decides if the plots are saved in the processed folder.
        fig_size: list
            2-D list to set the figure sizes. Default: [10,6]
        output_filter: bool
            Filters the output of excessive output.

        Returns
        -------
        dictionary
            Summary of the results.
        figures (optional)
            Bar plots of the specified results.

        """

        return smiles_analysis(self.smiles, path_to_export, download, plots, save_plots, fig_size, output_filter)

    def weight_vs_target_plot(self, target_name=None, save_fig = False, pre_standardization = True, path_to_export = None):
        """

        Parameters
        ----------
        target_name: str
            The title of the y-axis in the plot. Default: 'target'
        save_fig: bool
            Decides if the figure is saved in the processed directory.
        pre_standardization: bool
            Decides if the pre- or post-standardization target variable is used. Will only affect the scale,
            not the distribution. Default: True.
        path_to_export: str
            Export path, will default to the directory 'analysis_results' if not specified.

        Returns
        -------
        plot
            A seaborn jointplot of the molecular weight and target distributions.

        """

        target = self.raw_target if pre_standardization else self.target

        plot = mol_weight_vs_target(self.smiles, target, target_name=target_name, save_fig=save_fig,
                                    path_to_export=path_to_export)

        return plot


class GraphDataSet(DataSet):
    """A class that takes a path to a pickle file or a list of smiles and targets. The data is stored in
        Pytorch-Geometric Data instances and be accessed like an array. Additionally, if desired it splits the data and
        prepares the splits for training and validation. **If you wish to split the data immediately, please set
        split to True. Per default, it will not split.**

    Parameters
    ----------
    file_path: str
        The path to a pickle file that should be loaded and the data therein used.
    smiles: list of str
        List of smiles to be made into a graph.
    target: list of float
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list of str
        List of allowed atom symbols.
    atom_feature_list: list of str
        List of features to be applied. Default are the AFP atom features.
    bond_feature_list: list of str
        List of features that will be applied. Default are the AFP features
    split: bool
        An indicator if the dataset should be split. Only takes effect if nothing else regarding the split is specified
        and will trigger the default split. Default: False
    split_type: str
        Indicates what split should be used. Default: random. The options are:
        [consecutive, random, molecular weight, scaffold, stratified, custom]
    split_frac: list of float
        Indicates what the split fractions should be. Default: [0.8, 0.1, 0.1]
    custom_split: array
        The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
        where 0 indicates a training sample, 1 a testing sample and 2 a validation sample.
    log: bool
        Decides if the filtering output and other outputs will be shown.
    indices: np.array
        Can be used to override the indices of the data objects. Recommended not to use.


    """

    def __init__(self, file_path = None, smiles=None, target=None, global_features = None,
                 allowed_atoms = None, atom_feature_list = None, bond_feature_list = None, split = True,
                 split_type = None, split_frac = None, custom_split = None, log = False, indices=None):

        super().__init__(file_path=file_path, smiles=smiles, target=target, global_features=global_features,
                         allowed_atoms=allowed_atoms, atom_feature_list=atom_feature_list,
                         bond_feature_list=bond_feature_list, log=log, indices=indices)

        if split_type is None:
            split_type = 'random'
        else:
            split = True

        if split_frac is None:
            split_frac = [0.8, 0.1, 0.1]

        self.custom_split = custom_split
        self.split_type = split_type
        self.split_frac = split_frac


        assert np.sum(self.split_frac), 'Split fractions should add to 1.'

        if split or (self.custom_split is not None):
            self.train, self.test, self.val = split_data(data = self.data, split_type = self.split_type,
                                                        split_frac = self.split_frac, custom_split = self.custom_split)

    def get_splits(self, split_type = None, split_frac = None, custom_split = None):
        """ Returns the dataset split into training, testing and validation based on the given split type.

        Parameters
        ----------
        split_type: str
            Indicates what split should be used. It will either take a new argument or default
             to the initialized split fractions. The default initialization is 'random'. The options are:
             ['consecutive', 'random', 'molecular weight', 'scaffold', 'stratified', 'custom']
        split_frac: array
            Indicates what the split fractions should be. It will either take a new argument or default
             to the initialized split fractions. The default initialization is [0.8,0.1,0.1].
        custom_split: array
            The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
            where 0 indicates a training sample, 1 a testing sample and 2 a validation sample. It will either take
            a new argument of default to the initialized custom split. The default initialization is None.

        Returns
        -------
        train, test, val
            List containing the respective data objects.

        """

        split_type = self.split_type if split_type is None else split_type
        split_frac = self.split_frac if split_frac is None else split_frac
        custom_split = self.custom_split if custom_split is None else custom_split

        return split_data(data = self, split_type = split_type,
                            split_frac = split_frac, custom_split = custom_split)
