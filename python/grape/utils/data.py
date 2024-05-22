# Graph constructor built on top of pytorch geometric

import os
from collections.abc import Sequence
from typing import Union

import pickle

import pandas as pd
import numpy as np
from numpy import ndarray
import rdkit.Chem.Draw
import torch
from torch import tensor, Tensor

from rdkit import Chem
from rdkit.Chem import rdmolops, MolFromSmiles, Draw, MolToSmiles
from rdkit import RDLogger
import seaborn as sns


from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from grape.utils.featurizer import AtomFeaturizer, BondFeaturizer
#from grape.analysis import smiles_analysis
from grape.utils.split_utils import split_data
from grape.utils.feature_func import mol_weight

RDLogger.DisableLog('rdApp.*')

__all__ = ['filter_smiles',
           'construct_dataset',
           'DataLoad',
           'DataSet',
           'GraphDataSet']


##########################################################################
########### filtering ####################################################
##########################################################################

def filter_smiles(smiles: list[str], target: Union[list[str], list[float], ndarray], allowed_atoms: list[str] = None,
                  only_organic: bool = True, allow_dupes: bool = False, log: bool = False) -> Union[list,list]:
    """Filters a list of smiles based on the allowed atom symbols.

    Parameters
    ------------
    smiles: list of str
        Smiles to be filtered.
    target: list of int or list of float or ndarray
        Target of the graphs.
    allowed_atoms: list of str
        Valid atom symbols, non-valid symbols will be discarded. Default: [``C``, ``N``, ``O``, ``S``, ``F``, ``Cl``,
         ``Br``, ``I``, ``P``]
    only_organic: bool
        Checks if a molecule is ``organic`` counting the number of ``C`` atoms. If set to True, then molecules with less
        than one carbon will be discarded. Default: True
    log: bool
        Determines if there should be print-out statements to indicate why mols were filtered out. Default: False
    allow_dupes: bool
        Decides if duplicate smiles should be allowed. Default: False

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
                carbon_count = 0
                for atoms in mol.GetAtoms():
                    if atoms.GetSymbol() not in allowed_atoms:
                        if log:
                            print(f'SMILES {element} in index {list(df.smiles).index(element)} contains the atom {atoms.GetSymbol()} that is not'
                                f' permitted and will be ignored.')
                        indices_to_drop.append(list(df.smiles).index(element))
                    else:
                        if atoms.GetSymbol() == 'C':
                            carbon_count += 1

                if carbon_count < 1 and only_organic:
                    indices_to_drop.append(list(df.smiles).index(element))
                    if log:
                        print(f'SMILES {element} in index {list(df.smiles).index(element)} does not contain at least one'
                            f' carbon and will be ignored.')


    df.drop(indices_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # map smiles to mol and back to ensure SMILE notation consistency
    mols = df.smiles.map(lambda x: MolFromSmiles(x))
    df.smiles = mols.map(lambda x: MolToSmiles(x))

    if not allow_dupes:
        df.drop_duplicates(subset='smiles', inplace=True)

    return np.array(df.smiles), np.array(df.target)


##########################################################################
########### constructing a dataset #######################################
##########################################################################

def construct_dataset(smiles: list[str], target: Union[list[int], list[float], ndarray], allowed_atoms: list[str] = None,
                      atom_feature_list: list[str] = None, bond_feature_list: list[str] = None) -> list[Data]:
    """Constructs a dataset out of the smiles and target lists based on the feature lists provided. The dataset will be
    a list of torch geometric Data objects, using their conventions.

    Parameters
    ------------
    smiles : list of str
        Smiles that are featurized and passed into a PyG DataSet.
    target: list of int or list of float or ndarray
        Array of values that serve as the graph 'target'.
    allowed_atoms : list of str
        Smiles that are considered in featurization. Default: [``C``, ``N``, ``O``, ``S``, ``F``, ``Cl``, ``Br``,
        ``I``, ``P``]
    atom_feature_list : list of str
        Features of the featurizer, see utils.featurizer for more details. Default: All implemented features.

    bond_feature_list : list of str
        Bond features of the bond featurizer, see utils.featurizer for more details. Default: All implemented features.

    Returns
    --------
    list of Data
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
        data.append(Data(x = x,
                         edge_index = edge_index,
                         edge_attr = edge_attr,
                         y=tensor([target[i]], dtype=torch.float32)))

    return data


##########################################################################
########### Data classes #################################################
##########################################################################


class DataLoad(object):
    """The basic datasets-loading class. It holds the root as well as the raw  and processed files directories.

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
    """A class that takes a path to a pickle file or a list of smiles and targets. The datasets is stored in
    Pytorch-Geometric Data instances and be accessed like an array.

    Heavily inspired by the PyTorch-Geometric Dataset class, especially indices and index_select.

    Parameters
    ------------
    file_path: str
        The path to a pickle file that should be loaded and the datasets therein used.
    smiles: list of str
        List of smiles to be made into a graph.
    target: list of in or float
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list of str
        List of allowed atom symbols.
    only_organic: bool
        Checks if a molecule is ``organic`` counting the number of ``C`` atoms. If set to True, then molecules with less
        than one carbon will be discarded. Default: True
    atom_feature_list: list of str
        List of features to be applied. Default: All implemented features.
    bond_feature_list: list of str
        List of features that will be applied. Default: All implemented features.
    scale: bool
        Decides if the dataset should be scaled. Default: True
    log: bool
        Decides if the filtering output and other outputs will be shown.


    """
    def __init__(self, file_path: str = None, smiles: list[str] = None, target: Union[list[int], list[float],
    ndarray] = None, global_features:Union[list[float], ndarray] = None, filter: bool=True,allowed_atoms:list[str] = None,
    only_organic: bool = True, atom_feature_list: list[str] = None, bond_feature_list: list[str] = None,
    scale: bool = True, log: bool = False, root: str = None, indices:list[int] = None):

        assert (file_path is not None) or (smiles is not None and target is not None),'path or (smiles and target) must given.'

        super().__int__(root)

        if file_path is not None:
            with open(file_path, 'rb') as handle:
                try:
                    df = pd.read_pickle(handle)
                    print('Loaded dataset.')
                except:
                    raise ValueError('A dataset is stored as a DataFrame.')

            self.smiles = np.array(df.smiles)

            if scale:
                self.target, self.mean_target, self.std_target = self.standardize(np.array(df.target))

            self.global_features = np.array(df.global_features)
            self.data = list(df.graphs)

        else:
            if filter:
                self.smiles, self.raw_target = filter_smiles(smiles, target, allowed_atoms= allowed_atoms,
                                                     only_organic=only_organic, log=log)
            else:
                self.smiles, self.raw_target = np.array(smiles), np.array(target)

            # standardize target
            if scale:
                self.target, self.mean_target, self.std_target = self.standardize(np.array(self.raw_target))

            self.data = construct_dataset(smiles=self.smiles,
                                          target=self.target,
                                          allowed_atoms = allowed_atoms,
                                          atom_feature_list = atom_feature_list,
                                          bond_feature_list = bond_feature_list)

            self.global_features = global_features

        self._indices = indices
        self.num_node_features = self.data[0].num_node_features
        self.num_edge_features = self.data[0].num_edge_features
        self.data_name=None

        self.mol_weights = np.zeros(len(self.smiles))

        for i in range(len(self.smiles)):
            self.mol_weights[i] = mol_weight(Chem.MolFromSmiles(self.smiles[i]))

    @staticmethod
    def standardize(target):
        target_ = (target - np.mean(target)) / np.std(target)
        return target_, np.mean(target), np.std(target)

    @staticmethod
    def rescale(target, mean, std):
        return (target *std) + mean

    def scale_data(self):
        self.target, self.mean_target, self.std_target = self.standardize(np.array(self.raw_target))

    def rescale_data(self, target):
        return self.rescale(target, self.mean_target, self.std_target)

    def get_mol_weight(self):
        """Calculates the molecular weights of the DataSet smiles.

        Returns
        -------
        ndarray
            The molecular weight of the stored smiles.

        """
        weights = np.zeros(len(self.smiles))

        for i in range(len(self.smiles)):
            weights[i] = mol_weight(Chem.MolFromSmiles(self.smiles[i]))

        return weights



    def save_dataset(self, filename:str=None):
        """Loads the dataset (specifically the smiles, target, global features and graphs) to a pandas DataFrame and
        dumps it into a pickle file. Saving and loading the same dataset is about 20x faster than recreating it from
        scratch.

        Parameters
        ------------
        filename: str
            Name of the datasets file. Default: 'dataset'


        """
        if filename is None:
            if self.data_name is None:
                filename = 'dataset'
            else:
                filename = self.data_name

        path = self.processed_dir

        if not os.path.exists(path):
            os.makedirs(path)

        df_save = pd.DataFrame({'smiles':self.smiles,'target':self.raw_target,'global_features':self.global_features,
                   'graphs':self.data})

        path = os.path.join(path,filename+'.pickle')

        with open(path,'wb') as handle:
            pickle.dump(df_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'File saved at: {path}')


    def save_smiles(self, path: str =None):
        """Save the smiles as a text file in a given location.

        Parameters
        ------------
        path: str
            Path where the smiles should be saved. Default: The processed files' directory.

        """

        path = self.processed_dir if path is None else path

        if not os.path.exists(path):
            os.makedirs(path)

        np.savetxt(path+'/filtered_smiles.txt', X = np.array(self.smiles), fmt='%s')

    def get_mol(self) -> list[rdkit.Chem.Mol]:
        """Return a list containing all rdkit.Chem.Mol of the SMILES in the DataSet.

        Returns
        ---------
        list(rdkit.Chem.Mol)

        Example
        -------
        >>> from grape.datasets import BradleyDoublePlus
        >>> dataset = BradleyDoublePlus()
        >>> # Return the first 5 mol objects:
        >>> dataset.get_mol()[:5]
        [<rdkit.Chem.rdchem.Mol at 0x31cee26c0>,
         <rdkit.Chem.rdchem.Mol at 0x31cee2960>,
         <rdkit.Chem.rdchem.Mol at 0x31cee27a0>,
         <rdkit.Chem.rdchem.Mol at 0x31cee28f0>,
         <rdkit.Chem.rdchem.Mol at 0x31cee2a40>]

        """

        return list(map(lambda x: MolFromSmiles(x), self.smiles))

    def indices(self):
        """

        Returns
        ---------
        list
            Indices of the dataset

        """

        return range(len(self.data)) if self._indices is None else self._indices

    def __len__(self):
        """

        Returns
        ---------
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

    def index_select(self, idx:object):
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.

        Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset

        Parameters
        ----------
            idx: obj
                Index list of datasets objects to retrieve.

        Returns
        -------
        list
            Python list of datasets objects.

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

    def draw_smile(self, index: int)->rdkit.Chem.Draw.MolDraw2D:
        """Draw the smile at a desired index, best used in the jupyter environment.

        Parameters
        ----------
        index: int
            The index of the smile that should be drawn.

        Returns
        -------
        Image

        """
        return Draw.MolToImage(MolFromSmiles(self.smiles[index]))



    # def analysis(self, path_to_export:str = None, download:bool=False, plots:list = None, save_plots:bool = False,
    #              fig_size:tuple[int,int]=None, filter_output_txt:bool = True) -> tuple:
    #     """Returns an overview of different aspects of the smiles dataset **after filtering** according to:
    #     https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/utils/analysis.py.
    #     This includes the frequency of symbols, degree frequency and more.

    #     Parameters
    #     ----------
    #     path_to_export: str
    #         Path to the folder where analysis results should be saved. Default: None (recommended).
    #     download: bool
    #         Decides if the results are downloaded. If either the path is given or download is set to true, the
    #         analysis results will be downloaded as a txt file.
    #     plots: list of str
    #         Bar plots of the analysis results. Default: None. Possible options are:
    #         ['atom_type_frequency', 'degree_frequency', 'total_degree_frequency', 'explicit_valence_frequency',
    #         'implicit_valence_frequency', 'hybridization_frequency', 'total_num_h_frequency', 'formal_charge_frequency',
    #         'num_radical_electrons_frequency', 'aromatic_atom_frequency', 'chirality_tag_frequency',
    #         'bond_type_frequency', 'conjugated_bond_frequency', 'bond_stereo_configuration_frequency',
    #         'bond_direction_frequency']
    #     save_plots: bool
    #         Decides if the plots are saved in the processed folder.
    #     fig_size: list
    #         2-D list to set the figure sizes. Default: [10,6]
    #     filter_output_txt: bool
    #         Filters the output text file of some excessive information. Default: True
    #         .

    #     Returns
    #     -------
    #     dictionary
    #         Summary of the results.
    #     figures (optional)
    #         Bar plots of the specified results.

    #     """

    #     return smiles_analysis(self.smiles, path_to_export, download, plots, save_plots, fig_size, filter_output_txt)



    def get_splits(self, split_type:str = None, split_frac:list[float, float, float] = None,
                   custom_split:list[list] = None, **kwargs):
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
        SubSet, SubSet, SubSet
            List containing the respective datasets objects.

        """

        split_type = 'random' if split_type is None else split_type
        split_frac = [0.8, 0.1, 0.1] if split_frac is None else split_frac

        return split_data(data = self, split_type = split_type,
                            split_frac = split_frac, custom_split = custom_split, **kwargs)


class GraphDataSet(DataSet):
    """A class that takes a path to a pickle file or a list of smiles and targets. The dataset is stored in
        Pytorch-Geometric Data instances and be accessed like an array. Additionally, if desired it splits the datasets and
        prepares the splits for training and validation. **If you wish to split the datasets immediately, please set
        split to True. Per default, it will not split.**

    Parameters
    ----------
    file_path: str
        The path to a pickle file that should be loaded and the datasets therein used.
    smiles: list[str]
        List of smiles to be made into a graph.
    target: list[float]
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list[str]
        List of allowed atom symbols.
    atom_feature_list: list[str]
        List of features to be applied. Default are the AFP atom features.
    bond_feature_list: list[str]
        List of features that will be applied. Default are the AFP features
    split: bool
        An indicator if the dataset should be split. Only takes effect if nothing else regarding the split is specified
        and will trigger the default split. Default: False
    split_type: str
        Indicates what split should be used. Default: random. The options are:
        [consecutive, random, molecular weight, scaffold, stratified, custom]
    split_frac: list[float]
        Indicates what the split fractions should be. Default: [0.8, 0.1, 0.1]
    custom_split: list
        The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
        where 0 indicates a training sample, 1 a testing sample and 2 a validation sample.
    scale: bool
        Decides if the dataset should be scaled. Default: True
    log: bool
        Decides if the filtering output and other outputs will be shown.
    indices: list[int]
        Can be used to override the indices of the datasets objects. Recommended not to use.

    # TODO: Consider removing indices option
    """

    def __init__(self, file_path:str = None, smiles:list[str]=None, target: Union[list[int], list[float],
    ndarray] = None, global_features:Union[list[float], ndarray] = None,
    allowed_atoms:list[str] = None, only_organic: bool = True, atom_feature_list:list[str] = None,
    bond_feature_list:list[str] = None, split: bool = True, split_type:str = None, split_frac:list[float, float, float]
    = None, custom_split: list[int] = None, scale: bool = True, log: bool = False, indices:list[int] = None):

        super().__init__(file_path=file_path, smiles=smiles, target=target, global_features=global_features,
                         allowed_atoms=allowed_atoms, only_organic=only_organic,
                         atom_feature_list=atom_feature_list, scale=scale,
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
            self.train, self.val, self.test = split_data(data = self, split_type = self.split_type,
                                                        split_frac = self.split_frac, custom_split = self.custom_split)


    #def save_train_val_test_data(self, names: list[str]=None):
    #    assert len(names) == 3 or None, 'Names should have three elements: train, val, test'
