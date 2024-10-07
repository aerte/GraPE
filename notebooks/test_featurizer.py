from grape_chem.utils.featurizer import AtomFeaturizer, BondFeaturizer
import torch
from torch import tensor, Tensor
from rdkit import Chem
from rdkit.Chem import rdmolops, MolFromSmiles, Draw, MolToSmiles
from rdkit import RDLogger
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import dense_to_sparse

allowed_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']

atom_featurizer = AtomFeaturizer(allowed_atoms=allowed_atoms,
                                atom_feature_list = None)

bond_featurizer = BondFeaturizer(bond_feature_list=None)


data = []
smile = "CCCCCCCCCCCCCCCCCCCCCC(=O)O"
mol = MolFromSmiles(smile) # get into rdkit object
edge_index = dense_to_sparse(torch.tensor(rdmolops.GetAdjacencyMatrix(mol)))[0]
x = atom_featurizer(mol) #creates nodes
edge_attr = bond_featurizer(mol) #creates "edges" attrs
data_temp = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y=tensor([0],dtype=torch.float32))
# TODO: Might need to be tested on multidim global feats

data.append(data_temp)

print(data) #actual pyg graphs


