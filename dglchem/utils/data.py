# -*- coding: utf-8 -*-
#
# Graph constructor with input built on top of dgl-lifesci and pytorch geometric

import os

import pickle
import pandas as pd
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import rdmolops, MolFromSmiles
from rdkit import RDLogger

from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse

from dgllife.utils import splitters, analyze_mols


from dglchem.utils import featurizer

RDLogger.DisableLog('rdApp.*')

__all__ = ['filter_smiles',
           'construct_dataset',
           'DataSet',
           'MakeGraphDataSet']

def filter_smiles(smiles, target, allowed_atoms = None, print_out = False):
    """Filters a list of smiles based on the allowed atom symbols.

    Args
    ----------
    smiles: list of str
        Smiles to be filtered.
    target: list
        Target of the graphs.
    allowed_atoms: list of str
        Valid atom symbols, non-valid symbols will be discarded. Default: [``B``, ``C``, ``N``, ``O``,
            ``F``, ``Si``, ``P``, ``S``, ``Cl``, ``As``, ``Se``, ``Br``, ``Te``, ``I``, ``At``]
    print_out: bool
        Determines if there should be print-out statements to indicate why mols were filtered out. Default: False

    Returns
    ----------
    list[str]
        A list of filtered smiles strings.

    """

    if allowed_atoms is None:
        allowed_atoms = ['B', 'C', 'N', 'O','F', 'Si', 'P',
                       'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']

    df = pd.DataFrame({'smiles': smiles, 'target': target})
    indices_to_drop = []

    for element in smiles:
        mol = Chem.MolFromSmiles(element)

        if mol is None:
            if print_out:
                print(f'SMILES {element} in index {list(df.smiles).index(element)} is not valid.')
            indices_to_drop.append(list(df.smiles).index(element))

        else:
            if mol.GetNumHeavyAtoms() < 2:
                if print_out:
                    print(f'SMILES {element} in index {list(df.smiles).index(element)} consists of less than 2 heavy atoms'
                        f' and will be ignored.')
                indices_to_drop.append(list(df.smiles).index(element))

            else:
                for atoms in mol.GetAtoms():
                    if atoms.GetSymbol() not in allowed_atoms:
                        if print_out:
                            print(f'SMILES {element} in index {list(df.smiles).index(element)} contains the atom {atoms.GetSymbol()} that is not'
                                f' permitted and will be ignored.')
                        indices_to_drop.append(list(df.smiles).index(element))

    df.drop(indices_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return list(df.smiles), list(df.target)



def construct_dataset(smiles, target, allowed_atoms = None, atom_feature_list = None, bond_feature_list = None):
    """Constructs a dataset out of the smiles and target lists based on the feature lists provided.

    Args:
        smiles : list of str
            Smiles that are featurized and passed into a PyG DataSet.

        target: Any
            Array of values that serve as the graph 'target'.

        allowed_atoms : list of str
            Smiles that are considered in featurization. Default: [``B``, ``C``, ``N``, ``O``,
            ``F``, ``Si``, ``P``, ``S``, ``Cl``, ``As``, ``Se``, ``Br``, ``Te``, ``I``, ``At``,``other``]

        atom_feature_list : list of str
            Features of the featurizer, see utils.featurizer for more details. Default: AFP featurizer:
                atom_feature_list =
                    ['atom_type_one_hot','atom_degree_one_hot','atom_formal_charge',
                    'atom_num_radical_electrons',
                    'atom_hybridization_one_hot',
                    'atom_is_aromatic',
                    'atom_total_num_H_one_hot',
                    'atom_is_chiral_center',
                    'atom_chirality_type_one_hot']

        bond_feature_list : list of str
            Bond features of the bond featurizer, see utils.featurizer for more details. Default: AFP featurizer:

                bond_feats = ['bond_type_one_hot',
                          'bond_is_conjugated',
                          'bond_is_in_ring',
                          'bond_stereo_one_hot']

    Returns:
        data: Pytorch-Geometric DataSet object

    """
    if atom_feature_list is None:
        atom_feature_list = ['atom_type_one_hot', 'atom_degree_one_hot', 'atom_formal_charge',
         'atom_num_radical_electrons',
         'atom_hybridization_one_hot',
         'atom_is_aromatic',
         'atom_total_num_H_one_hot',
         'atom_is_chiral_center',
         'atom_chirality_type_one_hot']

    if bond_feature_list is None:
        bond_feature_list = ['bond_type_one_hot',
                      'bond_is_conjugated',
                      'bond_is_in_ring',
                      'bond_stereo_one_hot']

    atom_featurizer = featurizer.AtomFeaturizer(atom_data_field='node-feats', allowed_atoms=allowed_atoms,
                                                atom_feature_list = atom_feature_list)

    bond_featurizer = featurizer.BondFeaturizer(bond_data_field='edge-feats', bond_feature_list=bond_feature_list)


    data = []

    for (smile, i) in zip(smiles, range(len(smiles))):
        mol = MolFromSmiles(smile)
        edge_index = dense_to_sparse(torch.tensor(rdmolops.GetAdjacencyMatrix(mol)))[0]
        x = atom_featurizer(mol)['node-feats']
        edge_attr = bond_featurizer(mol)['edge-feats']
        data.append(Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y=target[i]))

    return data

class DataSet(object):
    """A class that takes a path to a pickle file or a list of smiles and targets. The data is stored in
        Pytorch-Geometric Data instances and be accessed like an array.

    Parameters
    ----------
    path: str
        The path to a pickle file that should be loaded and the data therein used.
    smiles: list of str
        List of smiles to be made into a graph.
    target: list of in or float
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list of str
        List of allowed atom symbols.
    atom_feature_list: list of str
        List of features to be applied. Default are the AFP atom features.
    bond_feature_list: list of str
        List of features that will be applied. Default are the AFP features
    log: bool
        Decides if the filtering output and other outputs will be shown.


    """
    def __init__(self, path = None, smiles = None, target = None, allowed_atoms = None,
                 atom_feature_list = None, bond_feature_list = None, log = False):

        assert (path is not None) or (smiles is not None and target is not None),'path or (smiles and target) must given.'

        if path is not None:
            with open(path, 'rb') as handle:
                self.data = pickle.load(handle)
        else:
            self.smiles, self.target = filter_smiles(smiles, target, allowed_atoms= allowed_atoms, print_out=log)

            self.data = construct_dataset(smiles=self.smiles,
                                          target=self.target,
                                          allowed_atoms = allowed_atoms,
                                          atom_feature_list = atom_feature_list,
                                          bond_feature_list = bond_feature_list)

    def save(self, filename, path=None):
        """Writes the dataset into a pickle file for easy reuse.

        Args:
            filename:
            path:

        Returns:

        """
        if path is None:
            path = os.getcwd()+'/data'
        if not os.path.exists(path):
            os.makedirs(path)
        print('Path to the saved file: ')

        with open(path+'/'+filename+'.pickle', 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'File saved at: {path}/{filename}.pickle')

    def __len__(self):
        """

        Returns: int
            Length of the dataset.

        """
        return len(self.data)

    def __getitem__(self, item):
        """

        Args:
            item: int
                Index of item to be returned.

        Returns: obj
            Data object at index [item].

        """

        return self.data[item]

    def analysis(self):
        """Returns an overview of different aspects of the smiles dataset **after filtering** according to:
        https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/utils/analysis.py.
        This includes the frequency of symbols, degree frequency and more.

        Returns: dict
            Dictionary with the analysis results.

        """

        return analyze_mols(self.smiles)




class MakeGraphDataSet(DataSet):
    """A class that takes a path to a pickle file or a list of smiles and targets. The data is stored in
        Pytorch-Geometric Data instances and be accessed like an array. Additionally, it splits the data and
        prepares the splits for training and validation. **If you do not wish to split the data immediately, please use
        the DataSet class instead.**

    Parameters
    ----------
    path: str
        The path to a pickle file that should be loaded and the data therein used.
    smiles: list of str
        List of smiles to be made into a graph.
    target: list of in or float
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list of str
        List of allowed atom symbols.
    atom_feature_list: list of str
        List of features to be applied. Default are the AFP atom features.
    bond_feature_list: list of str
        List of features that will be applied. Default are the AFP features
    log: bool
        Decides if the filtering output and other outputs will be shown.


    """

    def __init__(self, path = None, smiles=None, target=None, allowed_atoms = None,
                 atom_feature_list = None, bond_feature_list = None,
                 split_type = None, split_frac = None, log = False):

        super().__init__(path, smiles, target, allowed_atoms, atom_feature_list, bond_feature_list, log)


        if split_type is None:
            self.split_type = 'random'

        if split_frac is None:
            self.split_frac = [0.8, 0.1, 0.1]

        assert np.sum(self.split_frac), 'Split fractions should add to 1.'

        self.split_func = {
            'consecutive': splitters.ConsecutiveSplitter,
            'random': splitters.RandomSplitter,
            'molecular_weight': splitters.MolecularWeightSplitter,
            'scaffold': splitters.ScaffoldSplitter,
            'stratified': splitters.SingleTaskStratifiedSplitter
        }

        self.train, self.test, self.val = self.split_func[self.split_type].train_val_test_split(self.data,
                                                                                                self.split_frac[0],
                                                                                                self.split_frac[1],
                                                                                                self.split_frac[2])