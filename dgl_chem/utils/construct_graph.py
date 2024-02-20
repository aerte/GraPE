# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This code was adapted from https://github.com/awslabs/dgl-lifesci under the Apache-2.0 license.
#
# Graph constructer with input built on top of dgl-lifesci

from functools import partial
import torch
import dgl

from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops

from dgllife.utils import SMILESToBigraph
#from dgllife.utils import ToGraph

from dgl_chem.utils import featurizer

class GraphConstruct(object):
    """

    """

    def __init__(self,
                 allowed_atoms = None,
                 atom_feature_list = None,
                 bond_feature_list = None,
                 add_self_loop = None,
                 canonical_atom_order = True,
                 explicit_hydrogens = False,
                 num_virtual_nodes = 0):

        self.allowed_atoms = allowed_atoms
        self.atom_feats = atom_feature_list
        self.bond_feats = bond_feature_list
        self.add_loop = add_self_loop
        self.canonical_atom_order = canonical_atom_order
        self.explicit_hydrogens = explicit_hydrogens
        self.num_virtual_nodes = num_virtual_nodes

        # Atom featurizer:
        self.atom_featurizer = partial(featurizer.AtomFeaturizer, allowed_atoms = self.allowed_atoms,
                                  atom_feature_list = self.allowed_atoms)

        # Bond featurizer
        self.bond_featurizer = partial(featurizer.BondFeaturizer, bond_feature_list = self.bond_feats)

    def __call__(self, smiles):
        """

        Args:
            smiles:

        Returns:

        """
        return utils.SMILESToBigraph(smiles = smiles, )
