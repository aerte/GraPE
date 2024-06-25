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

# wrapper for the fragmentation
# TODO: make sure it imports properly

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

        motif_graph.nfeat = torch.Tensor(frag_features)
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
            motif_graph.feat = motif_edge_features
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph, mol)
            return frag_graph_list, motif_graph, atom_mask, frag_flag
        else:
            frag_graph_list = self.rebuild_frag_graph(frag_graph, motif_graph, mol)
            return frag_graph_list, motif_graph, atom_mask, frag_flag

    def compute_fragments(self, mol, graph, num_atoms):
        graph.edge_index = add_self_loops(graph.edge_index)[0] #might make it slower: TODO: investigate
        pat_list = []
        mol_size = mol.GetNumAtoms()
        for line in self.patterns:
            pat = Chem.MolFromSmarts(line[1])
            pat_list.append(list(mol.GetSubstructMatches(pat)))

        atom_idx_list = list(range(num_atoms))
        hit_ats = {}
        frag_flag = []
        prior_set = set()
        adj_masks = []
        atom_masks = []
        frag_features = []
        k = 0

        for idx, line in enumerate(self.patterns):
            pat_list = []
            mol_size = mol.GetNumAtoms()
            for line in self.patterns:
                pat = Chem.MolFromSmarts(line[1])
                pat_list.append(list(mol.GetSubstructMatches(pat)))

            atom_idx_list = [i for i in range(num_atoms)]
            hit_ats = {}
            frag_flag = []
            prior_set = set()
            k = 0

            for idx, line in enumerate(self.patterns):
                key = line[0]
                frags = pat_list[idx]
                if frags:
                    #print(f"in PyG got {len(frags)} frags")
                    for i, item in enumerate(frags):
                        item_set = set(item)
                        new_frags = frags[:i] + frags[i + 1:]
                        left_set = set(sum(new_frags, ()))
                        if not item_set.isdisjoint(left_set):
                            frags = new_frags

                    for frag in frags:
                        frag_set = set(frag)
                        if prior_set.isdisjoint(frag_set):
                            ats = frag_set
                        else:
                            ats = set() # /!\ dictionary? TODO: investigate
                        if ats:
                            adjacency_origin = Chem.rdmolops.GetAdjacencyMatrix(mol)[np.newaxis, :, :]
                            if k == 0:
                                adj_mask = adjacency_origin
                                atom_mask = torch.zeros((1, mol_size))
                                frag_features = torch.tensor([float(key == s) for s in self.frag_name_list], dtype=torch.float).unsqueeze(0)
                                #logger.debug("# in PyG #")
                                #logger.debug(adj_mask, "\n", atom_mask, "\n", frag_features)
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
                                # adj_mask = torch.cat((adj_mask, adjacency_origin), dim=0)
                                # atom_mask = torch.cat((atom_mask, torch.zeros((1, mol_size))), dim=0)
                                # frag_features = torch.cat((frag_features, torch.tensor([float(key == s) for s in self.frag_name_list], dtype=torch.float).unsqueeze(0)))
                                #logger.debug("# in PyG #")
                                #logger.debug(adj_mask, "\n", atom_mask, "\n", frag_features)
                            # if key not in hit_ats:
                            #     hit_ats[key] = torch.tensor(list(ats))
                            # else:
                            #     hit_ats[key] = torch.cat((hit_ats[key], torch.tensor(list(ats))))

                            ignores = list(set(atom_idx_list) - ats)
                            adj_mask[k, ignores, :] = 0
                            adj_mask[k, :, ignores] = 0 
                            atom_mask[k, list(ats)] = 1
                            frag_flag.append(key)
                            k += 1
                            prior_set.update(ats)

            # unknown fragments:
            unknown_ats = list(set(atom_idx_list) - prior_set)
            if len(unknown_ats) > 0:
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
                        hit_ats['unknown'] = np.vstack((hit_ats['unknown'], np.asarray(at)))
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
        num_motifs = motif_graph.num_nodes
        frag_graph_list = []
        #logging.debug("[PyG]\nnum_motifs: {}\n frag_graph edge index: {}\n frag_graph num nodes: {}\n\n".format(num_motifs, frag_graph.edge_index, frag_graph.num_nodes))
        for idx_motif in range(num_motifs):
            # Get the indices of nodes in this motif
            coord = motif_graph.atom_mask[idx_motif:idx_motif+1, :].nonzero()
            idx_list = []
            for idx_node in coord:
                idx_list.append(idx_node[1])
            # Create new fragment graph as a subgraph of the original
            new_frag_graph_conn = subgraph(idx_list, frag_graph.edge_index, relabel_nodes=True,)
            #TODO: make into new pyg Data object
            new_frag_graph= Data(
                edge_index=new_frag_graph_conn,

            )
            frag_graph_list.append(new_frag_graph)
        
        return frag_graph_list
    

def remove_edges(data, to_remove: list[tuple[int, int]]):
    """
    takes: PyG data object, list of pairs nodes making edges to remove
    returns: data with edges removed
    TODO: rewrite so that it also acts on:
    graph.feat
    graph.nfeat     (node features)      
    graph.atom_mask (optional)
    """
    edges_to_remove = []
    for src, tgt in to_remove:
        # we assume undirected graph (so both directions need to be added)
        idx = ((data.edge_index[0] == src) & (data.edge_index[1] == tgt)) | \
            ((data.edge_index[0] == tgt) & (data.edge_index[1] == src))
        edges_to_remove.append(idx.nonzero(as_tuple=True)[0])
    if (len(edges_to_remove) == 0):
        return data
    edges_to_remove = torch.cat(edges_to_remove)  # Flatten list of indices
    #logger.debug("Indices of Edges to Remove:\n", edges_to_remove)

    # True values : to edges we want to keep
    keep_edges = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    keep_edges[edges_to_remove] = False

    new_edge_index = data.edge_index[:, keep_edges]
    new_data = Data(x=data.x, edge_index=new_edge_index)
    return new_data