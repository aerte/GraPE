# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This code was adapted from https://github.com/awslabs/dgl-lifesci under the Apache-2.0 license.

import itertools

from _collections import defaultdict
from functools import partial

import numpy as np
import torch
import dgl.backend as F
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

__all__ = ['one_hot_encoding',
           'atom_type_one_hot',
           'atomic_number_one_hot',
           'atomic_number',
           'atom_degree_one_hot',
           'atom_degree',
           'atom_total_degree_one_hot',
           'atom_total_degree',
           'atom_explicit_valence_one_hot',
           'atom_explicit_valence',
           'atom_implicit_valence_one_hot',
           'atom_implicit_valence',
           'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot',
           'atom_total_num_H',
           'atom_formal_charge_one_hot',
           'atom_formal_charge',
           'atom_num_radical_electrons_one_hot',
           'atom_num_radical_electrons',
           'atom_is_aromatic_one_hot',
           'atom_is_aromatic',
           'atom_is_in_ring_one_hot',
           'atom_is_in_ring',
           'atom_chiral_tag_one_hot',
           'atom_chirality_type_one_hot',
           'atom_mass',
           'atom_is_chiral_center',
           'ConcatFeaturizer',
           'BaseAtomFeaturizer',
           'AtomFeaturizer',
           'bond_type_one_hot',
           'bond_is_conjugated_one_hot',
           'bond_is_conjugated',
           'bond_is_in_ring_one_hot',
           'bond_is_in_ring',
           'bond_stereo_one_hot',
           'bond_direction_one_hot',
           'BaseBondFeaturizer',
           'BondFeaturizer',
           ]


def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.

    Parameters
    ----------
    x
        Value to encode.
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element.

    Returns
    -------
    list
        List of boolean values where at most one value is True.
        The list is of length ``len(allowable_set)`` if ``encode_unknown=False``
        and ``len(allowable_set) + 1`` otherwise.

    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))

#################################################################
# Atom featurization
#################################################################

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Atom types to consider. Default: ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``,
        ``Cl``, ``Br``, ``Mg``, ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``,
        ``K``, ``Tl``, ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
        ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``, ``Cr``,
        ``Pt``, ``Hg``, ``Pb``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    """
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atomic_number_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the atomic number of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atomic numbers to consider. Default: ``1`` - ``100``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    """
    if allowable_set is None:
        allowable_set = list(range(1, 101))
    return one_hot_encoding(atom.GetAtomicNum(), allowable_set, encode_unknown)

def atomic_number(atom):
    """Get the atomic number for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
       List containing one int only.

    """
    return [atom.GetAtomicNum()]

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom degrees to consider. Default: ``0`` - ``10``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_degree(atom):
    """Get the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.
    """
    return [atom.GetDegree()]

def atom_total_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom including Hs.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list
        Total degrees to consider. Default: ``0`` - ``5``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    """
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)

def atom_total_degree(atom):
    """The degree of an atom including Hs.

    Returns
    -------
    list
        List containing one int only.
    """
    return [atom.GetTotalDegree()]

def atom_explicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the explicit valence of an aotm.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom explicit valences to consider. Default: ``1`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = list(range(1, 7))
    return one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)

def atom_explicit_valence(atom):
    """Get the explicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.
    """
    return [atom.GetExplicitValence()]

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom implicit valences to consider. Default: ``0`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_implicit_valence(atom):
    """Get the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Reurns
    ------
    list
        List containing one int only.
    """
    return [atom.GetImplicitValence()]

# pylint: disable=I1101
def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Total number of Hs to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

def atom_total_num_H(atom):
    """Get the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_num_H_one_hot
    """
    return [atom.GetTotalNumHs()]

def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the formal charge of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Formal charges to consider. Default: ``-2`` - ``2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_formal_charge
    """
    if allowable_set is None:
        allowable_set = list(range(-2, 3))
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    """Get formal charge for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_formal_charge_one_hot
    """
    return [atom.GetFormalCharge()]

def atom_partial_charge(atom):
    """Get Gasteiger partial charge for an atom.

    For using this function, you must have called ``AllChem.ComputeGasteigerCharges(mol)``
    to compute Gasteiger charges.

    Occasionally, we can get nan or infinity Gasteiger charges, in which case we will set
    the result to be 0.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one float only.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return [float(gasteiger_charge)]

def atom_num_radical_electrons_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the number of radical electrons of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Number of radical electrons to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetNumRadicalElectrons(), allowable_set, encode_unknown)

def atom_num_radical_electrons(atom):
    """Get the number of radical electrons for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.
    """
    return [atom.GetNumRadicalElectrons()]

def atom_is_aromatic_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.GetIsAromatic(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [atom.GetIsAromatic()]

def atom_is_in_ring_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.IsInRing(), allowable_set, encode_unknown)

def atom_is_in_ring(atom):
    """Get whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [atom.IsInRing()]

def atom_chiral_tag_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chiral tag of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.ChiralType
        Chiral tags to consider. Default: ``rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_OTHER``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List containing one bool only.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                         Chem.rdchem.ChiralType.CHI_OTHER]
    return one_hot_encoding(atom.GetChiralTag(), allowable_set, encode_unknown)

def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chirality type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Chirality types to consider. Default: ``R``, ``S``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List containing one bool only.
    """
    if not atom.HasProp('_CIPCode'):
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)

def atom_mass(atom, coef=0.01):
    """Get the mass of an atom and scale it.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    coef : float
        The mass will be multiplied by ``coef``.

    Returns
    -------
    list
        List containing one float only.
    """
    return [atom.GetMass() * coef]

def atom_is_chiral_center(atom):
    """Get whether the atom is chiral center

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [atom.HasProp('_ChiralityPossible')]

class ConcatFeaturizer(object):
    """Concatenate the evaluation results of multiple functions as a single feature.

    Parameters
    ----------
    func_list : list
        List of functions for computing molecular descriptors from objects of a same
        particular data type, e.g. ``rdkit.Chem.rdchem.Atom``. Each function is of signature
        ``func(data_type) -> list of float or bool or int``. The resulting order of
        the features will follow that of the functions in the list.
    """
    def __init__(self, func_list):
        self.func_list = func_list

    def __call__(self, x):
        """Featurize the input data.

        Parameters
        ----------
        x :
            Data to featurize.

        Returns
        -------
        list
            List of feature values, which can be of type bool, float or int.
        """
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers.

    Loop over all atoms in a molecule and featurize them with the ``featurizer_funcs``.

    **We assume the resulting DGLGraph will not contain any virtual nodes and a node i in the
    graph corresponds to exactly atom i in the molecule.**

    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Atom) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.
    """
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name]

    def __call__(self, mol):
        """Featurize all atoms in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features

class AtomFeaturizer(BaseAtomFeaturizer):
    """An atom featurizer based on a flexible input list.

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
    * **One hot encoding of an atoms chiral tag.**
      ---> *['atom_chiral_tag_one_hot']*
    * **Atomic mass.**
      ---> *['atom_mass']*
    * **If an atom is a chiral center (True/False).**
      ---> *['atom_is_chiral_center']*

    **The list argument order defines the feature order.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.
    allowed_atoms : list of str
        List of allowed atoms symbols. Default: [``B``, ``C``, ``N``, ``O``,
        ``F``, ``Si``, ``P``, ``S``, ``Cl``, ``As``, ``Se``, ``Br``, ``Te``, ``I``, ``At``,``other``]
    atom_feature_list: list of str
        List of features to be applied. Default are the AFP atom features:
            atom_feature_list =
                ['atom_type_one_hot','atom_degree_one_hot','atom_formal_charge',
                'atom_num_radical_electrons',
                'atom_hybridization_one_hot',
                'atom_is_aromatic',
                'atom_total_num_H_one_hot',
                'atom_is_chiral_center',
                'atom_chirality_type_one_hot']

    """
    def __init__(self, atom_data_field='h', allowed_atoms = None, atom_feature_list = None):

        if allowed_atoms is None:
            allowed_atoms = ['B', 'C', 'N', 'O', 'F', 'Si', 'P',
                           'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
        self.allowable_set_symbols = allowed_atoms

        ### Dictionary for all features
        total_atom_feat = {
        'atom_type_one_hot': partial(atom_type_one_hot, allowable_set = self.allowable_set_symbols, encode_unknown=True),
        'atomic_number_one_hot': atomic_number_one_hot,
        'atomic_number': atomic_number,
        'atom_degree_one_hot': partial(atom_degree_one_hot, allowable_set = list(range(6))),
        'atom_degree': atom_degree,
        'atom_total_degree_one_hot': atom_total_degree_one_hot,
        'atom_total_degree': atom_total_degree,
        'atom_explicit_valence_one_hot': atom_explicit_valence_one_hot,
        'atom_explicit_valence': atom_explicit_valence,
        'atom_implicit_valence_one_hot': atom_implicit_valence_one_hot,
        'atom_implicit_valence': atom_implicit_valence,
        'atom_hybridization_one_hot': partial(atom_hybridization_one_hot, encode_unknown = True),
        'atom_total_num_H_one_hot': atom_total_num_H_one_hot,
        'atom_total_num_H': atom_total_num_H,
        'atom_formal_charge_one_hot': atom_formal_charge_one_hot,
        'atom_formal_charge': atom_formal_charge,
        'atom_num_radical_electrons_one_hot': atom_num_radical_electrons_one_hot,
        'atom_num_radical_electrons': atom_num_radical_electrons,
        'atom_is_aromatic_one_hot': atom_is_aromatic_one_hot,
        'atom_is_aromatic': atom_is_aromatic,
        'atom_is_in_ring_one_hot': atom_is_in_ring_one_hot,
        'atom_is_in_ring': atom_is_in_ring,
        'atom_chiral_tag_one_hot': atom_chiral_tag_one_hot,
        'atom_chirality_type_one_hot': atom_chirality_type_one_hot,
        'atom_mass': atom_mass,
        'atom_is_chiral_center': atom_is_chiral_center,
        }


        feat_set = []
        for item in atom_feature_list:
            if item in total_atom_feat.keys():
                feat_set.append(total_atom_feat[item])
            else:
                print(f'Error: feature {item} is not an accepted feature.')
                continue

        super(AtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                feat_set
            )})


#################################################################
# Bond featurization
#################################################################

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondType
        Bond types to consider. Default: ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

def bond_is_conjugated_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the bond is conjugated.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.GetIsConjugated(), allowable_set, encode_unknown)

def bond_is_conjugated(bond):
    """Get whether the bond is conjugated.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [bond.GetIsConjugated()]

def bond_is_in_ring_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.IsInRing(), allowable_set, encode_unknown)

def bond_is_in_ring(bond):
    """Get whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [bond.IsInRing()]

def bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the stereo configuration of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of rdkit.Chem.rdchem.BondStereo
        Stereo configurations to consider. Default: ``rdkit.Chem.rdchem.BondStereo.STEREONONE``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOANY``, ``rdkit.Chem.rdchem.BondStereo.STEREOZ``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOE``, ``rdkit.Chem.rdchem.BondStereo.STEREOCIS``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOTRANS``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondStereo.STEREONONE,
                         Chem.rdchem.BondStereo.STEREOANY,
                         Chem.rdchem.BondStereo.STEREOZ,
                         Chem.rdchem.BondStereo.STEREOE,
                         Chem.rdchem.BondStereo.STEREOCIS,
                         Chem.rdchem.BondStereo.STEREOTRANS]
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)

def bond_direction_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the direction of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondDir
        Bond directions to consider. Default: ``Chem.rdchem.BondDir.NONE``,
        ``Chem.rdchem.BondDir.ENDUPRIGHT``, ``Chem.rdchem.BondDir.ENDDOWNRIGHT``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondDir.NONE,
                         Chem.rdchem.BondDir.ENDUPRIGHT,
                         Chem.rdchem.BondDir.ENDDOWNRIGHT]
    return one_hot_encoding(bond.GetBondDir(), allowable_set, encode_unknown)

class BaseBondFeaturizer(object):
    """An abstract class for bond featurizers.
    Loop over all bonds in a molecule and featurize them with the ``featurizer_funcs``.
    We assume the constructed ``DGLGraph`` is a bi-directed graph where the **i** th bond in the
    molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the **(2i)**-th and **(2i+1)**-th edges
    in the DGLGraph.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Bond) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops in each bond feature.
        The features of the self loops will be zero except for the additional columns.
    """
    def __init__(self, featurizer_funcs, feat_sizes=None, self_loop=False):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes
        self._self_loop = self_loop

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        mol = Chem.MolFromSmiles('CCO')
        feats = self(mol)

        return feats[feat_name].shape[1]

    def __call__(self, mol):
        """Featurize all bonds in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        if self._self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
                self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
                self_loop_feats[:, -1] = 1
                feats = torch.cat([feats, self_loop_feats], dim=0)
                processed_features[feat_name] = feats

        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles('CO')
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.zeros(num_atoms, feats.shape[1])
                feats[:, -1] = 1
                processed_features[feat_name] = feats

        return processed_features


class BondFeaturizer(BaseBondFeaturizer):
    """A bond featurizer based on a flexible input list.

    All possible bond features are:

    * **One hot encoding of the bond type.**:
      ---> *['bond_type_one_hot']*
    * **One hot encoding if bond is conjugated (true/false).**:
      ---> *['bond_is_conjugated_one_hot']*
    * **Bond conjugation (true/false).**:
      ---> *['bond_is_conjugated']*
    * **One hot encoding if bond is in a ring**:
      ---> *['bond_is_in_ring_one_hot']*
    * **Bond in a ring (true/false)**:
      ---> *['bond_is_in_ring']*
    * **One hot encoding of bond stereo configuration.**:
      ---> *['bond_stereo_one_hot']*
    * **One hot encoding of bond direction.**:
      ---> *['bond_direction_one_hot']*

    **The input feature list determines the order of the features.**

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops. The feature of the
        self loops will be zero except for the additional column.
    bond_feature_list: list of str
        List of features that will be applied. Default are the AFP features:
            bond_feats = ['bond_type_one_hot',
                          'bond_is_conjugated',
                          'bond_is_in_ring',
                          'bond_stereo_one_hot']
    """
    def __init__(self, bond_data_field='e', self_loop=False, bond_feature_list=None):

        if bond_feature_list is None:
            bond_feature_list = ['bond_type_one_hot',
                                 'bond_is_conjugated',
                                 'bond_is_in_ring',
                                 'bond_stereo_one_hot']

        total_bond_feats = {
            'bond_type_one_hot': bond_type_one_hot,
            'bond_is_conjugated_one_hot': bond_is_conjugated_one_hot,
            'bond_is_conjugated': bond_is_conjugated,
            'bond_is_in_ring_one_hot': bond_is_in_ring_one_hot,
            'bond_is_in_ring': bond_is_in_ring,
            'bond_stereo_one_hot':partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                             Chem.rdchem.BondStereo.STEREOANY,
                                                             Chem.rdchem.BondStereo.STEREOZ,
                                                             Chem.rdchem.BondStereo.STEREOE]),
            'bond_direction_one_hot':bond_direction_one_hot
        }

        feat_set = []
        for item in bond_feature_list:
            if item in total_bond_feats:
                feat_set.append(total_bond_feats[item])
            else:
                print(f'Error: Bond feature {item} is not an accepted feature.')
                continue

        super(BondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                feat_set
            )}, self_loop=self_loop)