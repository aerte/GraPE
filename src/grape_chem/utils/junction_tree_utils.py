##########################################################################################
#####################    JT CODE START  ##################################################
##########################################################################################
##########################################################################################
#####################    JT CODE START  ##################################################
##########################################################################################
##########################################################################################
#####################    JT CODE START  ##################################################
##########################################################################################


import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch_geometric.data import Data

from rdkit import Chem

from torch_geometric.utils import subgraph
from torch_geometric.utils import add_self_loops

import scipy.sparse as sp

# temp imports for debugging function
import matplotlib.pyplot as plt
import networkx as nkx
from torch_geometric.utils import to_networkx

# temporary import
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

__all__ =[
    "graph_2_frag",
    "JT_SubGraph"
]

def graph_2_frag(smiles, origin_graph, JT_subgraph):
    mol = Chem.MolFromSmiles(smiles)
    frag_graph_list, motif_graph, atom_mask, frag_flag = JT_subgraph.fragmentation(origin_graph, mol)
    return frag_graph_list, motif_graph, atom_mask, frag_flag

def find_edge_ids(edge_index, src_nodes, dst_nodes):
    # This assumes edge_index is 2 * num_edges
    edge_ids = []
    for src, dst in zip(src_nodes, dst_nodes):
        # Find indices where the source and destination match the provided nodes
        mask = (edge_index[0] == src) & (edge_index[1] == dst)
        edge_ids.extend(mask.nonzero(as_tuple=False).squeeze(1).tolist())
    return edge_ids

def add_edge(data, edge):
    """
    Add an edge to a PyTorch Geometric graph.
    Takes:
        data (torch_geometric.data.Data): The graph data object.
        edge (tuple) (src, dst)
    Returns:
        torch_geometric.data.Data: The updated graph data object with the new edge added.
    """
    # get the source and target nodes
    idx1, idx2 = edge
    new_edge = torch.tensor([[idx1], [idx2]], dtype=torch.long)
    
    if data.edge_index.device != new_edge.device:
        new_edge = new_edge.to(data.edge_index.device)
    
    data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
    return data

def visualize_pyg_graph(data, mol):
    """
    Helper function can be sometimes useful for debugging PyG stuff
    does not actually respect node information so don't rely on it
    for debugging the fragmentation algo.
    """
    G = to_networkx(data, to_undirected=True, remove_self_loops=1)
    plt.figure(figsize=[15, 7])
    plt.title(Chem.MolToSmiles(mol))
    options = {
        'node_color': 'black',
        'node_size': 20,
        'width': 1,
    }
    nkx.draw(G, with_labels=True, **options)
    plt.show()

# use "MG_plus_reference" as scheme

###############################################################################
###############                 Junction tree                  ################
###############################################################################

class JT_SubGraph(object):
    def __init__(self, scheme, save_file_path=None, verbose=True):
        path = os.path.join('./env', scheme + '.csv') # change to your needs TODO: load from yaml or larger config of script where called
        data_from = os.path.realpath(path)
        df = pd.read_csv(data_from)
        pattern = df[['First-Order Group', 'SMARTs', 'Priority']].values.tolist()
        self.patterns = sorted(pattern, key=lambda x: x[2], reverse=False)
        self.frag_name_list = [x[0] for x in self.patterns]
        self.frag_dim = len(self.frag_name_list)
        self.save_file_path = save_file_path
        self.verbose = verbose

    def fragmentation(self, graph, mol, check_metadata=False):
        """
        Parameters:
        - graph: The input graph to fragment.
        - mol: The RDKit molecule object.
        - save_file_path: Optional; path to save/load the fragmentation result.
        Currently that logic is implemented in the `_prepare_frag` method of DataSet class (TODO: change this)
        - check_metadata: Optional; if True, checks fragment metadata before returning a loaded file.
        
        Returns:
        - frag_graph_list: List of fragment graphs (subgraphs resulting of fragmentation).
        - motif_graph: The "motif graph" (junction tree), encoding connectivity between fragments
        - atom_mask: for each fragment, a mask of atoms in the original molecule.
        - frag_flag: Fragment flags identifying fragments to nodes in the motif graph.
        """
        num_atoms = mol.GetNumAtoms()

        frag_graph, frag_flag, atom_mask, idx_tuples, frag_features = self.compute_fragments(mol, graph, num_atoms)
        num_motifs = atom_mask.shape[0]

        edge_index = torch.tensor([], dtype=torch.long)
        motif_graph = Data(edge_index=edge_index,)
        motif_graph.num_nodes = num_motifs
        _, idx_tuples, motif_graph = self.build_adjacency_motifs(atom_mask, idx_tuples, motif_graph)

        if frag_features.ndim == 1:
            frag_features = frag_features.reshape(-1, 1).transpose()

        motif_graph.x = torch.Tensor(frag_features)
        motif_graph.atom_mask = torch.Tensor(atom_mask)

        # if necessary
        edge_features = graph.edge_attr  # Assuming edge_attr stores the features
        add_edge_feats_ids_list = []

        for _, item in enumerate(idx_tuples): #TODO: maybe needs error handling?
            es = find_edge_ids(graph.edge_index, [item[0]], [item[1]])   
            add_edge_feats_ids_list.append(es)
        add_edge_feats_ids_list[:] = [i for sublist in add_edge_feats_ids_list for i in sublist]   
        
        if num_atoms != 1:
            # Assuming a procedure to handle the features as necessary
            motif_edge_features = edge_features[add_edge_feats_ids_list, :] #da same
            motif_graph.edge_attr = motif_edge_features
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph, mol)
        else:
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph, mol)

        return frag_graph_list, motif_graph, atom_mask, frag_flag
    

    def compute_fragments(self, mol, graph, num_atoms):
        clean_edge_index = graph.edge_index
        # graph.edge_index = add_self_loops(graph.edge_index)[0] # might make it slower: TODO: investigate #this part changes the self loops
        pat_list = []
        mol_size = mol.GetNumAtoms()
        num_atoms = mol.GetNumAtoms()
        for line in self.patterns:
            pat = Chem.MolFromSmarts(line[1])
            pat_list.append(list(mol.GetSubstructMatches(pat)))
            #if pat_list[-1] != []:
                #print("Pattern: ", line, " found in molecule")
        atom_idx_list = list(range(num_atoms))
        hit_ats = {}
        frag_flag = [] # List[str], len := #fragments
        prior_set = set()
        adj_mask = []
        atom_mask = []
        frag_features = []
        k = 0

        for idx, line in enumerate(self.patterns):
            key = line[0]
            frags = pat_list[idx]
            #print(frags)
                #remove all the nodes in the frag that might appear multiple times until they appear 
            for i, item in enumerate(frags):
                item_set = set(item) #set(int)
                new_frags = frags[:i] + frags[i + 1:]
                left_set = set(sum(new_frags, ()))
                if not item_set.isdisjoint(left_set):
                    frags = new_frags

            for frag in frags: #frag:tuple in frags:List[Tuples]
                frag_set = set(frag)
                if not prior_set.isdisjoint(frag_set) or not frag_set:
                    continue
                ats = frag_set
                adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)
                adj_mask.append(adjacency_origin.copy())
                atom_mask.append(torch.zeros((mol_size,)))
                frag_features.append(torch.tensor([float(key == s) for s in self.frag_name_list], dtype=torch.float))

                if key not in hit_ats.keys():
                    hit_ats[key] = np.asarray(list(ats))
                else:
                    hit_ats[key] = np.vstack((hit_ats[key], np.asarray(list(ats))))
                ignores = list(set(atom_idx_list) - set(ats))
                adj_mask[k][ignores, :] = 0
                adj_mask[k][:, ignores] = 0 
                atom_mask[k][list(ats)] = 1
                frag_flag.append(key)
                k += 1
                prior_set.update(ats)

        # unknown fragments:
        unknown_ats = list(set(atom_idx_list) - prior_set)
        for i, at in enumerate(unknown_ats):
            if k == 0:
                if num_atoms == 1:
                    adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)
                adj_mask = adjacency_origin
                atom_mask = np.zeros((1, mol_size))
            else:
                # adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(m)[np.newaxis, :, :]
                adj_mask.append(adjacency_origin.copy())
                atom_mask.append(torch.zeros((mol_size,)))
            if 'unknown' not in hit_ats.keys():
                hit_ats['unknown'] = np.asarray(at)
            else:
                hit_ats['unknown'] = np.append(hit_ats['unknown'], np.asarray(at)) #stack all unknown atoms into 1 thing
            ignores = list(set(atom_idx_list) - set([at]))

            if num_atoms != 1:
                adj_mask[k][ignores, :] = 0
                adj_mask[k][:, ignores] = 0

            atom_mask[k][at] = 1
            frag_flag.append('unknown')
            if num_atoms != 1:
                frag_features.append(np.asarray(list(map(lambda s: float('unknown' == s), self.frag_name_list))))
            else:
                frag_features = np.asarray(list(map(lambda s: float('unknown' == s), self.frag_name_list))) #convert to PyG
            k += 1
            #should be modified to only vstack at the end instead of in all the complex conditions
        #### end of preprocessing #####

        if k > 0:
            frag_features = np.asarray(frag_features)
            adj_mask = np.asarray(adj_mask)
            atom_mask = np.asarray(atom_mask)
        
        adjacency_fragments = adj_mask.sum(axis=0)

        idx1, idx2 = (adjacency_origin - adjacency_fragments).nonzero()
  
        idx_tuples = list(zip(idx1.tolist(), idx2.tolist())) # the tuples to remove?
        #if bigraph is wanted it should be setup here
        frag_graph = remove_edges(graph, idx_tuples)
        graph.edge_index = clean_edge_index #set the edge index back. Quick fix TODO: find a better way to count self loops instead
        return frag_graph, frag_flag, atom_mask, idx_tuples, frag_features
        
    def build_adjacency_motifs(self, atom_mask, idx_tuples, motif_graph):
        k = atom_mask.shape[0]
        duplicate_bond = []
        adjacency_motifs = np.zeros((k, k)).astype(int)
        motif_edge_begin = list(map(lambda x: self.atom_locate_frag(atom_mask, x[0]), idx_tuples))
        motif_edge_end = list(map(lambda x: self.atom_locate_frag(atom_mask, x[1]), idx_tuples))

        # eliminate duplicate bond in triangle substructure
        for idx1, idx2 in zip(motif_edge_begin, motif_edge_end):
            if adjacency_motifs[idx1, idx2] == 0:
                adjacency_motifs[idx1, idx2] = 1
                add_edge(motif_graph, (idx1, idx2))
            else:
                rm_1 = self.frag_locate_atom(atom_mask, idx1)
                rm_2 = self.frag_locate_atom(atom_mask, idx2)
                if isinstance(rm_1, int):
                    rm_1 = [rm_1]
                if isinstance(rm_2, int):
                    rm_2 = [rm_2]
                for i in rm_1:
                    for j in rm_2:
                        duplicate_bond.extend([tup for tup in idx_tuples if tup == (i, j)])
        if duplicate_bond:
            idx_tuples.remove(duplicate_bond[0])
            idx_tuples.remove(duplicate_bond[2])
        return adjacency_motifs, idx_tuples, motif_graph

    def atom_locate_frag(self, atom_mask, atom):
        return atom_mask[:, atom].tolist().index(1)

    def frag_locate_atom(self, atom_mask, frag):
        return atom_mask[frag, :].nonzero()[0].tolist()

    def rebuild_frag_graph(self, frag_graph, motif_graph, mol=None):
        if frag_graph.x is None:
            print("FRAG GRAPF X IS NONE !!!")
        num_motifs = motif_graph.num_nodes
        frag_graph_list = []

        for idx_motif in range(num_motifs):
            # Get the indices of nodes in this motif
            coord = motif_graph.atom_mask[idx_motif:idx_motif+1, :].nonzero(as_tuple=True)[1]
            idx_list = coord.tolist()

            # Create new fragment graph as a subgraph of the original
            new_graph_edge_index, new_graph_edge_attr = subgraph(
                idx_list, frag_graph.edge_index, edge_attr=frag_graph.edge_attr, relabel_nodes=True, num_nodes=frag_graph.num_nodes,
            )

            new_node_features = frag_graph.x[idx_list] if frag_graph.x is not None else None

            new_frag_graph = Data(
                edge_index=new_graph_edge_index,
                edge_attr=new_graph_edge_attr,
                num_nodes=len(idx_list), 
                x=new_node_features #explicitly passing nodes. TODO: unit test to make sure feats match with origin graph
            )
            frag_graph_list.append(new_frag_graph)
        
        return frag_graph_list


def remove_edges(data, to_remove: list[tuple[int, int]]):
    """
    Takes: PyG data object, list of pairs of nodes making edges to remove.
    Returns: Data with specified edges removed, including edge attributes.
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    
    # List to store indices of edges to keep
    keep_indices = []

    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        
        if (src, tgt) not in to_remove and (tgt, src) not in to_remove: #removes both directions
            keep_indices.append(i)
    

    keep_indices = torch.tensor(keep_indices, dtype=torch.long)
    #filter edges and attr over mask
    new_edge_index = edge_index[:, keep_indices]
    new_edge_attr = edge_attr[keep_indices] if edge_attr is not None else None
    
    new_data = Data(x=data.x, edge_index=new_edge_index, edge_attr=new_edge_attr, num_nodes=data.num_nodes)
    return new_data
##########################################################################################
#####################    JT CODE END    ##################################################
##########################################################################################
##########################################################################################
#####################    JT CODE END    ##################################################
##########################################################################################
##########################################################################################
#####################    JT CODE END    ##################################################
##########################################################################################

def remove_edges_other(data, to_remove: list[tuple[int, int]]):
    """
    Other more PyG-esque take on the remove edges function
    TODO: unit test against top function
    """
    edge_indices = []
    for src, tgt in to_remove:
        edge_indices.extend((data.edge_index[0] == src) & (data.edge_index[1] == tgt)) #idea from ChatGPT
        edge_indices.extend((data.edge_index[0] == tgt) & (data.edge_index[1] == src))

    if not edge_indices:
        return data

    mask = ~torch.tensor(edge_indices)
    data.edge_index = data.edge_index[:, mask] # fails
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[mask]

    return data