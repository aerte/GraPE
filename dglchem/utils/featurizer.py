# Atom and Bond featurizers
#
# Inspired by https://github.com/awslabs/dgl-lifesci

import itertools

from _collections import defaultdict
from functools import partial

import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

# feature extraction functions from dgl life
from dgllife.utils import featurizers as ft

__all__ = [
    'AtomFeaturizer',
    'BondFeaturizer'
]

class AtomFeaturizer(object):
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

    **Virtual nodes still have to be implemented.**

    Parameters
    ----------
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
    def __init__(self, allowed_atoms = None, atom_feature_list = None):

        if allowed_atoms is None:
            allowed_atoms = ['B', 'C', 'N', 'O', 'F', 'Si', 'P',
                           'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']

        self.allowable_set_symbols = allowed_atoms

        ### Dictionary for all features
        total_atom_feat = {
        'atom_type_one_hot': partial(ft.atom_type_one_hot, allowable_set = self.allowable_set_symbols, encode_unknown=False),
        'atomic_number_one_hot': ft.atomic_number_one_hot,
        'atomic_number': ft.atomic_number,
        'atom_degree_one_hot': partial(ft.atom_degree_one_hot, allowable_set = list(range(6))),
        'atom_degree': ft.atom_degree,
        'atom_total_degree_one_hot': ft.atom_total_degree_one_hot,
        'atom_total_degree': ft.atom_total_degree,
        'atom_explicit_valence_one_hot': ft.atom_explicit_valence_one_hot,
        'atom_explicit_valence': ft.atom_explicit_valence,
        'atom_implicit_valence_one_hot': ft.atom_implicit_valence_one_hot,
        'atom_implicit_valence': ft.atom_implicit_valence,
        'atom_hybridization_one_hot': partial(ft.atom_hybridization_one_hot, encode_unknown = False),
        'atom_total_num_H_one_hot': ft.atom_total_num_H_one_hot,
        'atom_total_num_H': ft.atom_total_num_H,
        'atom_formal_charge_one_hot': ft.atom_formal_charge_one_hot,
        'atom_formal_charge': ft.atom_formal_charge,
        'atom_num_radical_electrons_one_hot': ft.atom_num_radical_electrons_one_hot,
        'atom_num_radical_electrons': ft.atom_num_radical_electrons,
        'atom_is_aromatic_one_hot': ft.atom_is_aromatic_one_hot,
        'atom_is_aromatic': ft.atom_is_aromatic,
        'atom_is_in_ring_one_hot': ft.atom_is_in_ring_one_hot,
        'atom_is_in_ring': ft.atom_is_in_ring,
        'atom_chiral_tag_one_hot': ft.atom_chiral_tag_one_hot,
        'atom_chirality_type_one_hot': ft.atom_chirality_type_one_hot,
        'atom_mass': ft.atom_mass,
        'atom_is_chiral_center': ft.atom_is_chiral_center,
        }

        if atom_feature_list is None:
            atom_feature_list = list(total_atom_feat.keys())
        self.atom_feature_list = atom_feature_list

        self.feat_set = []
        for item in atom_feature_list:
            if item in total_atom_feat.keys():
                self.feat_set.append(total_atom_feat[item])
            else:
                print(f'Error: feature {item} is not an accepted feature.')
                continue

    def __call__(self, mol):
        num_atoms = mol.GetNumAtoms()
        atom_feats = dict()

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


class BondFeaturizer(object):
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

    Parameters
    ----------
    bond_feature_list: list of str
        List of features that will be applied. Default are the AFP features:
            bond_feats = ['bond_type_one_hot',
                          'bond_is_conjugated',
                          'bond_is_in_ring',
                          'bond_stereo_one_hot']
    """
    def __init__(self, bond_feature_list=None):

        total_bond_feats = {
            'bond_type_one_hot': ft.bond_type_one_hot,
            'bond_is_conjugated_one_hot': ft.bond_is_conjugated_one_hot,
            'bond_is_conjugated': ft.bond_is_conjugated,
            'bond_is_in_ring_one_hot': ft.bond_is_in_ring_one_hot,
            'bond_is_in_ring': ft.bond_is_in_ring,
            'bond_stereo_one_hot':partial(ft.bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                             Chem.rdchem.BondStereo.STEREOANY,
                                                             Chem.rdchem.BondStereo.STEREOZ,
                                                             Chem.rdchem.BondStereo.STEREOE]),
            'bond_direction_one_hot':ft.bond_direction_one_hot
        }

        if bond_feature_list is None:
            bond_feature_list = list(total_bond_feats.keys())

        self.bond_feature_list = bond_feature_list

        self.feat_set = []

        for item in self.bond_feature_list:
            if item in total_bond_feats:
                self.feat_set.append(total_bond_feats[item])
            else:
                print(f'Error: Bond feature {item} is not an accepted feature.')
                continue

    def __call__(self, mol):
        num_bonds = mol.GetNumBonds()
        bond_feats = dict()

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
