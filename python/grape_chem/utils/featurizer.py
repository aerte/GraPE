# Atom and Bond featurizers
#
# Inspired by the feature extractor of  https://github.com/awslabs/dgl-lifesci

import itertools

from _collections import defaultdict
from functools import partial

from typing import Callable, Union

import numpy as np
import rdkit.Chem
import torch
from rdkit import Chem

from grape_chem.utils import feature_func as ff


__all__ = [
    'AtomFeaturizer',
    'BondFeaturizer'
]

class FunctionNotWellDefined(Exception):
    pass

##########################################################################
########### Atom featurizer ##############################################
##########################################################################

class AtomFeaturizer(object):
    """An atom featurizer based on a flexible input list. Inspired by https://github.com/awslabs/dgl-lifesci.

    The possible features include:

    * **One hot encoding of atoms types.**
      ---> *['atom_type_one_hot']*: based on a list of allowed atoms.
    * **One hot encoding of atomic numbers.**
      ---> *['atomic_number_one_hot']*
    * **Atomic number of atoms.**
      ---> *['atomic_number']*
    * **One hot encoding of atom degree.**
      ---> *['atom_degree_one_hot']*
    * **Atom degree.**
      ---> *['atom_degree']*
    * **One hot encoding of atom degree (including H).**
      ---> *['atom_total_degree_one_hot']*
    * **Atom degree (including H).**
      ---> *['atom_total_degree']*
    * **One hot encoding of the atom valence (explicit/including H).**
      ---> *['atom_explicit_valence_one_hot']*
    * **Atom valence (explicit/including H).**
      ---> *['atom_explicit_valence']*
    * **One hot encoding of the atom valence (implicit).**
      ---> *['atom_implicit_valence_one_hot']*
    * **Atom valence (implicit).**
      ---> *['atom_implicit_valence']*
    * **One hot encoding of the atom hybridization.**
      ---> *['atom_hybridization_one_hot']*
    * **One hot encoding of the total number of H.**
      ---> *['atom_total_num_H_one_hot']*
    * **Number of H.**
      ---> *['atom_total_num_H']*
    * **One hot encoding og the atom formal charge.**
      ---> *['atom_formal_charge_one_hot']*
    * **Formal charge.**
      ---> *['atom_formal_charge']*
    * **One hot encoding of the number of radical electrons of an atom.**
      ---> *['atom_num_radical_electrons_one_hot']*
    * **Number of radical electrons of an atom.**
      ---> *['atom_num_radical_electrons']*
    * **One hot encoding whether an atom is aromatic.**
      ---> *['atom_is_aromatic_one_hot']*
    * **If an atom is aromatic (True/False).**
      ---> *['atom_is_aromatic']*
    * **One hot encoding whether an atom is in a ring.**
      ---> *['atom_is_in_ring_one_hot']*
    * **If an atom is in a ring (True/False).**
      ---> *['atom_is_in_ring']*
    * **One hot encoding of an atoms chiral tag**
      ---> *['atom_chiral_tag_one_hot']*
    * **Atomic mass.**
      ---> *['atom_mass']*
    * **If an atom is a chiral center (True/False).**
      ---> *['atom_is_chiral_center']*

    **The list argument order defines the feature order.

    **Virtual nodes still have to be implemented.**

    ----

    References

    [1] Adem R.N. Aouichaoui et al., Application of interpretable group-embedded graph neural networks for pure compound
    properties, 2023, https://doi.org/10.1016/j.compchemeng.2023.108291

    [2] Adem R.N. Aouichaoui et al., Combining Group-Contribution Concept and Graph Neural Networks Toward Interpretable
    Molecular Property Models, 2023, https://doi.org/10.1021/acs.jcim.2c01091

    -----

    Parameters
    -----------
    allowed_atoms : list of str
        List of allowed atoms symbols. The default follows the choice of atom symbols allowed in [1, 2]. Default:
        [``C``, ``N``, ``O``, ``S``, ``F``, ``Cl``, ``Br``, ``I``, ``P``].
    atom_feature_list: list of str
        List of features to be applied. Default are the features used in [1,2], which are: [``atom_type_one_hot``,
        ``atom_total_num_H_one_hot``, ``atom_total_degree_one_hot``, ``atom_explicit_valence_one_hot``,
        ``atom_hybridization_one_hot``, ``atom_is_aromatic``, ``atom_is_chiral_center``, ``atom_chirality_type_one_hot``,
        ``atom_chiral_tag_one_hot``, ``atom_formal_charge``].

    Example
    ---------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from grape_chem.utils import AtomFeaturizer

    >>> mol = MolFromSmiles('CO')
    >>> atom_featurizer = AtomFeaturizer(['atom_type_one_hot', 'atom_total_degree_one_hot'])
    >>> atom_featurizer(mol)
    tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    >>> # We see in the output that C has 3 more bonds than O, which matches our expectation.

    """
    def __init__(self, atom_feature_list: list[str] = None, allowed_atoms: list[str] = None):

        self.allowable_set_symbols = allowed_atoms

        ### Dictionary for all features
        self.total_atom_feat = {
        'atom_type_one_hot': ff.atom_symbol,
        'atomic_number_one_hot': partial(ff.atom_number, type_='one_hot'),
        'atomic_number': ff.atom_number,
        'atom_degree_one_hot': partial(ff.atom_degree, type_='one_hot'),
        'atom_degree': ff.atom_degree,
        'atom_total_degree_one_hot': partial(ff.atom_degree, type_='total_one_hot', encode_unknown=True),
        'atom_total_degree': partial(ff.atom_degree, type_='total'),
        'atom_explicit_valence_one_hot': partial(ff.atom_valence, type_='ex_one_hot', encode_unknown=True),
        'atom_explicit_valence': partial(ff.atom_valence, type_='ex'),
        'atom_implicit_valence_one_hot': partial(ff.atom_valence, type_='im_one_hot', encode_unknown=True),
        'atom_implicit_valence': ff.atom_valence,
        'atom_hybridization_one_hot': ff.atom_hybridization,
        'atom_total_num_H_one_hot': partial(ff.atom_num_H, type_='one_hot'),
        'atom_total_num_H': ff.atom_num_H,
        'atom_formal_charge_one_hot': partial(ff.atom_formal_charge, type_='one_hot'),
        'atom_formal_charge': ff.atom_formal_charge,
        'atom_num_radical_electrons_one_hot': partial(ff.atom_num_rad_electrons, type_='one_hot'),
        'atom_num_radical_electrons': ff.atom_num_rad_electrons,
        'atom_is_aromatic': ff.atom_is_aromatic,
        'atom_is_in_ring': ff.atom_in_ring,
        'atom_chiral_tag_one_hot': partial(ff.atom_chiral, type_='tag'),
        'atom_chirality_type_one_hot': partial(ff.atom_chiral, type_='type'),
        'atom_is_chiral_center': partial(ff.atom_chiral, type_='center'),
        'atom_mass': ff.atom_mass,
        }

        if atom_feature_list is None:
            atom_feature_list = [
                'atom_type_one_hot',
                'atom_total_num_H_one_hot',
                'atom_total_degree_one_hot',
                'atom_explicit_valence_one_hot',
                'atom_hybridization_one_hot',
                'atom_is_aromatic',
                'atom_is_chiral_center',
                'atom_chirality_type_one_hot',
                'atom_chiral_tag_one_hot',
                'atom_formal_charge'
            ]

        self.atom_feature_list = atom_feature_list

        self.feat_set = []
        for item in atom_feature_list:
            if item in self.total_atom_feat.keys():
                self.feat_set.append(self.total_atom_feat[item])
            else:
                print(f'Error: feature {item} is not an accepted feature.')
                continue

    def __call__(self, mol: rdkit.Chem.Mol):
        num_atoms = mol.GetNumAtoms()
        atom_feats = defaultdict()

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)

            atom_feats[i] = []

            for func_name, func in zip(self.atom_feature_list, self.feat_set):
                atom_feats[i].append(func(atom))

        feats = []

        for atom_num in atom_feats.keys():
            feats.append(np.concatenate(atom_feats[atom_num]))
        feats = np.array(feats)

        return torch.tensor(feats, dtype=torch.float32)

    def extend_features(self, func_names: list[str], funcs: list[Callable]):
        """Adds the possibility of extending the featurizers function dictionary with new functions. The additional functions
        must take a RDKit Atom as input! For other features, please use global features of the Graph DataSet (fx. if you
        wanted to add the total molecular weight). While it tests if the function output type is
        valid, it cannot test the true validity!

        Parameters
        -----------
        func_names: list of str
            Function names that will be the keys of the dictionary input.
        funcs: list of Callables
            Functions that will be executed in the featurizer if called in the feature_list or by default.

        Example
        ---------
        >>> from grape_chem.utils import AtomFeaturizer
        >>> from rdkit.Chem import MolFromSmiles

        >>> # Add a function that returns True if the molecular weight of an atom is above 15[u]
        >>> def add_atom_mass_above_15(atom):
        >>>     if atom.GetMass() > 15:
        >>>         return [True]
        >>>     else:
        >>>         return [False]
        >>> featurizer = AtomFeaturizer(['atom_type_one_hot'])
        >>> featurizer.extend_features(['atom_mass_above_15'],[add_atom_mass_above_15])
        >>> featurizer.atom_feature_list
        ['atom_type_one_hot','atom_mass_above_30']
        >>> featurizer(MolFromSmiles('COO'))
        Testing the function with C atom. The output is: [False]
        tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 1.]])
        >>> # We see that carbon did not fulfill the condition, while the two oxygen atoms did (as we expected).


        """

        for name, func in zip(func_names, funcs):
            try:
                test = func(Chem.MolFromSmiles('COO').GetAtomWithIdx(0))
                print(f'Testing the function with C atom. The output is: {test}')
            except:
                raise FunctionNotWellDefined('Function to be added does not work.')

            assert isinstance(test, list), 'The added function output must be a one-dimensional list.'
            assert np.array(test).ndim == 1, 'The added function must have one dimensional output'

            try:
                np.sum(np.array(test))
            except:
                raise TypeError('The function must return only number or booleans')

            self.atom_feature_list.append(name)
            self.feat_set.append(func)




##########################################################################
########### Bond featurizer ##############################################
##########################################################################


class BondFeaturizer(object):
    """A bond featurizer based on a flexible input list. Inspired by https://github.com/awslabs/dgl-lifesci.

    All possible bond features are:

    * **One hot encoding of the bond type**.
      ---> *['bond_type_one_hot']*
    * **Bond conjugation (true/false)**.
      ---> *['bond_is_conjugated']*
    * **Bond in a ring (true/false)**.
      ---> *['bond_is_in_ring']*
    * **One hot encoding of bond stereo configuration**.
      ---> *['bond_stereo_one_hot']*
    * **One hot encoding of bond direction**.
      ---> *['bond_direction_one_hot']*

    **The input feature list determines the order of the features. All bonds are saved bidirectionally (twice) by
    default to adhere to the PyTorch Geometric structure.**

    ------

    References

    [1] Adem R.N. Aouichaoui et al., Application of interpretable group-embedded graph neural networks for pure compound
    properties, 2023, https://doi.org/10.1016/j.compchemeng.2023.108291

    [2] Adem R.N. Aouichaoui et al., Combining Group-Contribution Concept and Graph Neural Networks Toward Interpretable
    Molecular Property Models, 2023, https://doi.org/10.1021/acs.jcim.2c01091

    -----

    Parameters
    ------------
    bond_feature_list: list of str
        List of features that will be applied. Default: The features used in [1,2], which are: [``bond_type_one_hot``,
        ``bond_is_conjugated``, ``bond_is_in_ring``,``bond_stereo_one_hot``]

    Example
    ---------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from grape_chem.utils import BondFeaturizer

    >>> mol = MolFromSmiles('Clc1cccs1')
    >>> bond_featurizer = BondFeaturizer(['bond_type_one_hot','bond_is_in_ring'])
    >>> bond_featurizer(mol)
    tensor([[1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.]])
    >>> # We see that the first chlorine and carbon are connected via a single bond which is not part of the ring,
    while all the remaining bonds are part of a ring.

    References
    ----------
    [1] Adem R.N. Aouichaoui et al., Application of interpretable group-embedded graph neural networks for pure compound
    properties, 2023, https://doi.org/10.1016/j.compchemeng.2023.108291
    [2] Adem R.N. Aouichaoui et al., Combining Group-Contribution Concept and Graph Neural Networks Toward Interpretable
    Molecular Property Models, 2023, https://doi.org/10.1021/acs.jcim.2c01091

    """
    def __init__(self, bond_feature_list: list[str] =None):

        total_bond_feats = {
            'bond_type_one_hot': ff.bond_type,
            'bond_is_conjugated': ff.bond_is_conjugated,
            'bond_is_in_ring': ff.bond_in_ring,
            'bond_stereo_one_hot': ff.bond_stereo,
            'bond_direction_one_hot':ff.bond_direction
        }

        if bond_feature_list is None:
            bond_feature_list = [
                'bond_type_one_hot',
                'bond_is_conjugated',
                'bond_is_in_ring',
                'bond_stereo_one_hot'
            ]

        self.bond_feature_list = bond_feature_list

        self.feat_set = []

        for item in self.bond_feature_list:
            if item in total_bond_feats:
                self.feat_set.append(total_bond_feats[item])
            else:
                print(f'Error: Bond feature {item} is not an accepted feature.')
                continue

    def __call__(self, mol: rdkit.Chem.Mol):
        num_bonds = mol.GetNumBonds()
        bond_feats = defaultdict()

        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)

            bond_feats[i] = []

            for func_name, func in zip(self.bond_feature_list, self.feat_set):
                bond_feats[i].append(func(bond))

        feats = []

        for bond_num in bond_feats.keys():
            # Has to be saved twice
            feats.append(np.concatenate(bond_feats[bond_num]))
            feats.append(np.concatenate(bond_feats[bond_num]))
        feats = np.array(feats)

        return torch.tensor(feats, dtype=torch.float32)

    def extend_features(self, func_names: list[str], funcs: list[Callable]):
        """Adds the possibility of extending the featurizers function dictionary with new functions. The additional functions
        must take a RDKit Bond as input! For other features, please use global features of the Graph DataSet (fx. if you
        wanted to add the total molecular weight). While it tests if the function output type is valid, it cannot test
        the true validity!

        Parameters
        -----------
        func_names: list
            Function names that will be the keys of the dictionary input.
        funcs: list
            Functions that will be executed in the featurizer if called in the feature_list or by default.

        """

        for name, func in zip(func_names, funcs):
            try:
                test = func(Chem.MolFromSmiles('COO').GetBondWithIdx(0))
                print(f'Testing the function CO bond. The output is: {test}')
            except:
                raise FunctionNotWellDefined('Function to be added does not work.')

            assert isinstance(test, list), 'The added function output must be a one-dimensional list.'
            assert np.array(test).ndim == 1, 'The added function must have one dimensional output'

            try:
                np.sum(np.array(test))
            except:
                raise TypeError('The function must return only number or booleans')

            self.atom_feature_list.append(name)
            self.feat_set.append(func)
