import unittest
import torch
from torch_geometric.data import Data
from frag_graphs_compare import wanted_frag_graphs #to unclutter
from grape_chem.utils.junction_tree_utils import remove_edges #JT_SubGraph
from grape_chem.utils.data import construct_dataset
from rdkit import Chem

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
import torch
from torch_geometric.data import Data
from rdkit import Chem

import numpy as np

from torch_geometric.utils import subgraph
from torch_geometric.utils import add_self_loops

import scipy.sparse as sp

#stinky-ass imports
import matplotlib.pyplot as plt
import dgl
import networkx as nkx
from torch_geometric.utils import to_networkx

#temporary import
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

def visualize_dgl_graph(g, mol):
    options = {
        'node_color': 'black',
        'node_size': 20,
        'width': 1,
    }
    G = dgl.to_networkx(g)
    plt.figure(figsize=[15,7])
    plt.title(Chem.MolToSmiles(mol))
    nkx.draw(G, **options)
    plt.show()

def visualize_pyg_graph(data, mol):
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

#TODO: refactor, very messy

def dgl_to_pyg(dgl_graph):
    """
    Convert a DGL graph to a PyTorch Geometric graph.
    Parameters:
        dgl_graph (dgl.DGLGraph): the DGL graph to convert
    Returns:
        torch_geometric.data.Data: the graph encoded in PyG format
    made by chatGPT so don't trust it
    """
    # Get node features from DGL graph
    if dgl_graph.ndata:
        x = {key: dgl_graph.ndata[key] for key in dgl_graph.ndata}
    else:
        x = None

    # Get edge features from DGL graph
    if dgl_graph.edata:
        edge_attr = {key: dgl_graph.edata[key] for key in dgl_graph.edata}
    else:
        edge_attr = None
    
    # Get edge indices from DGL graph
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    
    # Convert edge indices to the same dtype and device as the node features if necessary
    if x is not None:
        first_key = next(iter(x))
        edge_index = edge_index.to(x[first_key].device).type(x[first_key].dtype)

    pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return pyg_graph

#use "MG_plus_reference" as scheme

###############################################################################
###############                 Junction tree                  ################
###############################################################################

class JT_SubGraph(object):
    def __init__(self, scheme):
        path = os.path.join('./env', scheme + '.csv') #change to your needs TODO: load from yaml or larger config of script where called
        data_from = os.path.realpath(path)
        df = pd.read_csv(data_from)
        pattern = df[['First-Order Group', 'SMARTs', 'Priority']].values.tolist()
        self.patterns = sorted(pattern, key=lambda x: x[2], reverse=False)
        self.frag_name_list = [x[0] for x in self.patterns]
        self.frag_dim = len(self.frag_name_list)

    def fragmentation(self, graph, mol):
        """
        TODO: write real desc
        while converting, good to know:
            graph.edata['feat'] is graph.feat
            motif_graph.ndata['feat'] is graph.nfeat          (arbitrary)
            motif_graph.ndata['atom_mask'] is graph.atom_mask (arbitrary)
        we assume that the ndata['feat'] from before entering the function is not needed
        """
    
        #graph = from_dgl(graph) #if passing a dgl for debugging
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
            return frag_graph_list, motif_graph, atom_mask, frag_flag
        else:
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph, mol)
            return frag_graph_list, motif_graph, atom_mask, frag_flag

    def compute_fragments(self, mol, graph, num_atoms):
        clean_edge_index = graph.edge_index
        #graph.edge_index = add_self_loops(graph.edge_index)[0] #might make it slower: TODO: investigate #this part changes the self loops
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
        adj_masks = []
        atom_masks = []
        frag_features = []
        k = 0

        for idx, line in enumerate(self.patterns):
            key = line[0]
            frags = pat_list[idx]
            #print(frags)
            if frags:
                #remove all the nodes in the frag that might appear multiple times until they appear 
                for i, item in enumerate(frags):
                    item_set = set(item) #set(int)
                    new_frags = frags[:i] + frags[i + 1:]
                    left_set = set(sum(new_frags, ()))
                    if not item_set.isdisjoint(left_set):
                        frags = new_frags

                for frag in frags: #frag:tuple in frags:List[Tuples]
                    frag_set = set(frag)
                    if prior_set.isdisjoint(frag_set):
                        ats = frag_set
                    else:
                        ats = set() # /!\ dictionary? TODO: investigate
                    if ats: #same condition as if set is disjoint
                        adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)[np.newaxis, :, :]
                        if k == 0:
                            adj_mask = adjacency_origin
                            atom_mask = torch.zeros((1, mol_size))
                            frag_features = torch.tensor([float(key == s) for s in self.frag_name_list], dtype=torch.float).unsqueeze(0)
                        else:
                            adj_mask = np.vstack((adj_mask, adjacency_origin))
                            atom_mask = np.vstack((atom_mask, np.zeros((1, mol_size))))
                            frag_features = np.vstack((frag_features,
                                                    np.asarray(
                                                        list(map(lambda s: float(key == s), self.frag_name_list)))))
                        if key not in hit_ats.keys():
                            hit_ats[key] = np.asarray(list(ats))
                        else:
                            hit_ats[key] = np.vstack((hit_ats[key], np.asarray(list(ats))))
                        ignores = list(set(atom_idx_list) - set(ats))

                        adj_mask[k, ignores, :] = 0
                        adj_mask[k, :, ignores] = 0 
                        atom_mask[k, list(ats)] = 1
                        frag_flag.append(key)
                        k += 1
                        prior_set.update(ats)

        # unknown fragments:
        unknown_ats = list(set(atom_idx_list) - prior_set)
        if len(unknown_ats) > 0: #useless
            for i, at in enumerate(unknown_ats):
                if k == 0:
                    if num_atoms == 1:
                        adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)[np.newaxis, :, :]
                    adj_mask = adjacency_origin
                    atom_mask = np.zeros((1, mol_size))
                else:
                    # adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(m)[np.newaxis, :, :]
                    adj_mask = np.vstack((adj_mask, adjacency_origin))
                    atom_mask = np.vstack((atom_mask, np.zeros((1, mol_size))))
                if 'unknown' not in hit_ats.keys():
                    hit_ats['unknown'] = np.asarray(at)
                else:
                    hit_ats['unknown'] = np.vstack((hit_ats['unknown'], np.asarray(at))) #stack all unknown atoms into 1 thing
                ignores = list(set(atom_idx_list) - set([at]))
                # print(prior_idx)
                if num_atoms != 1:
                    adj_mask[k, ignores, :] = 0
                    adj_mask[k, :, ignores] = 0
                atom_mask[k, at] = 1
                frag_flag.append('unknown')
                if num_atoms != 1:
                    frag_features = np.vstack( #convert to PyG
                        (frag_features, np.asarray(list(map(lambda s: float('unknown' == s), self.frag_name_list)))))
                else:
                    frag_features = np.asarray(list(map(lambda s: float('unknown' == s), self.frag_name_list))) #convert to PyG
                k += 1
                #should be modified to only vstack at the end instead of in all the complex conditions
        #### end of preprocessing #####
        adjacency_fragments = adj_mask.sum(axis=0)
        try:
            idx1, idx2 = (adjacency_origin.squeeze(0) - adjacency_fragments).nonzero()
        except:
            breakpoint()  
        # idx_tuples: list of tuples, idx of begin&end atoms on each new edge
        idx_tuples = list(zip(idx1.tolist(), idx2.tolist())) # the tuples to remove?
        # remove reverse edges
        # idx_tuples = list(set([tuple(sorted(item)) for item in idx_tuples]))

        frag_graph = remove_edges(graph, idx_tuples)
        graph.edge_index = clean_edge_index #set the edge index back. Quick fix TODO: find a better way to count self loops instead
        return frag_graph, frag_flag, atom_mask, idx_tuples, frag_features
        
    def build_adjacency_motifs(self, atom_mask, idx_tuples, motif_graph):
        k = atom_mask.shape[0]
        duplicate_bond = []
        adjacency_motifs = np.zeros((k, k)).astype(int)
        motif_edge_begin = list(map(lambda x: self.atom_locate_frag(atom_mask, x[0]), idx_tuples))
        motif_edge_end = list(map(lambda x: self.atom_locate_frag(atom_mask, x[1]), idx_tuples))
        #adjacency_motifs[new_edge_begin, new_edge_end] = 1
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
            #breakpoint()
            coord = motif_graph.atom_mask[idx_motif:idx_motif+1, :].nonzero(as_tuple=True)[1]
            idx_list = coord.tolist()
            # idx_list = [] #keeping in case need to revert
            # for idx_node in coord:
            #     idx_list.append(idx_node[1])
            # Create new fragment graph as a subgraph of the original
            new_graph_edge_index, new_graph_edge_attr,= subgraph(
                idx_list, frag_graph.edge_index, edge_attr=frag_graph.edge_attr, relabel_nodes=True,num_nodes=frag_graph.num_nodes,
            )

            new_node_features = frag_graph.x[idx_list] if frag_graph.x is not None else None

            new_frag_graph= Data(
                edge_index=new_graph_edge_index,
                edge_attr=new_graph_edge_attr,
                num_nodes=len(idx_list), 
                x=new_node_features #explicitely passing nodes. TODO: unit test to make sure feats match with origin graph
            )
            frag_graph_list.append(new_frag_graph)
        
        return frag_graph_list

##########################################################################################
#####################    JT CODE END    ##################################################
##########################################################################################
##########################################################################################
#####################    JT CODE END    ##################################################
##########################################################################################
##########################################################################################
#####################    JT CODE END    ##################################################
##########################################################################################


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

class TestRemoveEdges(unittest.TestCase):
    def setUp(self):
        """
        creates dummy data test doubles
        """
        self.edge_index = torch.tensor([[0, 1, 2, 3],
                                        [1, 2, 3, 0]])
        self.x = torch.tensor([[1], [2], [3], [4]])
        self.edge_attr = torch.tensor([[1], [2], [3], [4]])
        self.num_nodes = 4
        self.data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes, edge_attr=self.edge_attr)

    def test_remove_edges_only_removes_wanted_edges(self):
        to_remove = [(0, 1), (1, 2)]
        updated_data = remove_edges(self.data, to_remove)

        expected_edge_index = torch.tensor([[2, 3],
                                            [3, 0]])
        expected_edge_attr = torch.tensor([[3], [4]])

        self.assertTrue(torch.equal(updated_data.edge_index, expected_edge_index))
        self.assertTrue(torch.equal(updated_data.edge_attr, expected_edge_attr))

    def test_remove_edges_only_acts_on_edges(self):
        """
        sounds silly but is needed when AI tools are involved
        """
        to_remove = [(0, 1)]
        updated_data = remove_edges(self.data, to_remove)

        self.assertTrue(torch.equal(updated_data.x, self.data.x))
        self.assertEqual(updated_data.num_nodes, self.data.num_nodes)

    def test_remove_edges_also_acts_on_edge_attr(self):
        to_remove = [(1, 2)]
        updated_data = remove_edges(self.data, to_remove)

        expected_edge_index = torch.tensor([[0, 2, 3],
                                            [1, 3, 0]])
        expected_edge_attr = torch.tensor([[1], [3], [4]])

        self.assertTrue(torch.equal(updated_data.edge_index, expected_edge_index))
        self.assertTrue(torch.equal(updated_data.edge_attr, expected_edge_attr))

    def test_remove_edges_leaves_data_unchanged_when_no_edges_to_remove(self):
        to_remove = []
        updated_data = remove_edges(self.data, to_remove)

        self.assertTrue(torch.equal(updated_data.edge_index, self.data.edge_index))
        self.assertTrue(torch.equal(updated_data.edge_attr, self.data.edge_attr))
        self.assertTrue(torch.equal(updated_data.x, self.data.x))
        self.assertEqual(updated_data.num_nodes, self.data.num_nodes)

class TestJTSubGraph(unittest.TestCase):
    def setUp(self):
        """
        will be using the molecule c1ccccc1(C(=O)C) as a test case
        from the paper `https://pubs.acs.org/doi/10.1021/acs.jcim.2c01091?ref=pdf`
        code from paper, which produced the attributes of the graph we use for testing, is available at 
        `https://github.com/gsi-lab/GC-GNN/blob/38741b8364c29fb1271ced47ada85bc3e8c6e686/utils/junctiontree_encoder.py`,
        commit `ea932e9`.
        We compare our fragmentation to the DGL encoding of graphs produced by that code.
        """
        #using molecule from 
        self.smiles = "c1ccccc1(C(=O)C)"
        self.mol = Chem.MolFromSmiles(self.smiles)

        #TODO: move to test case 
        self.wanted_num_nodes = 9
        self.wanted_num_edges = 4
        self.wanted_node_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.wanted_edge_indices_dgl = {
            (1, 2),
            (2, 1),
            (1, 3),
            (3, 1),
        }
        self.wanted_node_features = {
            'feat': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            'global_feat': [[9.0, 120.15099999999995, 0.0, 1.0], [9.0, 120.15099999999995, 0.0, 1.0], [9.0, 120.15099999999995, 0.0, 1.0], [9.0, 120.15099999999995, 0.0, 1.0], [9.0, 120.15099999999995, 0.0, 1.0], [9.0, 120.15099999999995, 0.0, 1.0], [9.0, 120.15099999999995, 0.0, 1.0], [9.0, 120.15099999999995, 0.0, 1.0], [9.0, 120.15099999999995, 0.0, 1.0]],
        }
        self.wanted_edge_features = {
            'feat': [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        }

        self.wanted_frag_graphs_list = wanted_frag_graphs

        # (Pdb) atom_mask
        self.wanted_atom_mask=[
                [0., 1., 1., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 0., 0., 0., 0.]
            ]
        self.wanted_frag_flag=['ACC=O', 'ACH', 'ACH', 'ACH', 'ACH', 'ACH', 'CH3']

        # init JT_SubGraph instance
        fragmentation_scheme = "MG_plus_reference" #make sure your env/ folder is set up from the data files in the repo (notebooks/data_files)
        self.jt_subgraph = JT_SubGraph(scheme=fragmentation_scheme) 

        graph = construct_dataset([self.smiles], None, graph_only=True)
        self.frag_graph_list, self.motif_graph, self.atom_mask, self.frag_flag = self.jt_subgraph.fragmentation(graph, self.mol)


    def test_jt_subgraph_produces_consistent_data(self):
        self.assertTrue(len(self.frag_graph_list) == len(self.atom_mask)  == len(self.frag_flag))

    def test_jt_subgraph_produces_well_formed_graphs(self):
        graph = construct_dataset([self.smiles], None, graph_only=True)
        frag_graph_list, motif_graph, atom_mask, frag_flag = self.jt_subgraph.fragmentation(graph, self.mol)
        for frag_graph in frag_graph_list:
            self.assertTrue(frag_graph.num_nodes == frag_graph.x.shape[0])
            self.assertTrue(frag_graph.edge_index.shape[1] == frag_graph.edge_attr.shape[0])
    
    def test_jt_subgraph_produces_correct_number_frags(self):
        self.assertTrue(len(self.frag_graph_list) == len(self.wanted_frag_graphs_list))

    def test_jt_subgraph_produces_correct_num_nodes(self): 
        for i in range(len(self.frag_graph_list)): #TODO: refactor to use zip
            self.assertTrue(self.frag_graph_list[i].num_nodes == self.wanted_frag_graphs_list[i]['num_nodes'])

    def test_jt_subgraph_produces_correct_num_edges(self): 
        for i in range(len(self.frag_graph_list)): #TODO: refactor to use zip
            self.assertTrue(self.frag_graph_list[i].edge_index.shape[1] == self.wanted_frag_graphs_list[i]['num_edges'] == len(self.wanted_frag_graphs_list[i]['edge_indices']))

    def test_jt_subgraph_edges_match_with_other_implementation(self):
        """
        PyG graphs store edges in COO format with an array of src and an array of dest indexed
        in the same way. DGL produces a list of tuples of edges. Our graphs are undirected so we
        manipulate them before comparing.
        """
        coo_src = self.frag_graph_list[0].edge_index[0].tolist()
        coo_dst = self.frag_graph_list[0].edge_index[1].tolist()
        coo_as_tupleset = {(min(s, d), max(s, d)) for s, d in zip(coo_src, coo_dst)}
        tuples_as_tupleset = {(min(a, b), max(a, b)) for a, b in wanted_frag_graphs[0]['edge_indices']}

        # Compare sets of edges
        self.assertEqual(coo_as_tupleset, tuples_as_tupleset, "Edge lists do not match for undirected graph")
if __name__ == '__main__':
    unittest.main()



from rdkit import Chem

# SMILES strings
# smiles1 = 'CC(=O)C1=CC=CC=C1'
# smiles2 = 'c1ccccc1(C(=O)C)'
# mol1 = Chem.CanonSmiles(smiles1)
# mol2 = Chem.CanonSmiles(smiles2)
