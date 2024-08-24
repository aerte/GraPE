import unittest
import torch
from torch_geometric.data import Data
from frag_graphs_compare import wanted_frag_graphs #to unclutter
#from grape_chem.utils.junction_tree_utils import remove_edges, JT_SubGraph
from old_jt_encoder import remove_edges, JT_SubGraph
from grape_chem.utils.data import construct_dataset
from rdkit import Chem
import numpy as np
from numpy.testing import assert_array_equal
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

    def test_remove_edges_removes_both_directions(self):
        #TODO: implement
        pass 

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
        in the same way. DGL produces a list of tuples of edges.
        ->The node indexing between the two implementations is not the same so for now this fails...
        """
        pass
        if False:
            for i in range(len(self.frag_graph_list)):
                coo_src = self.frag_graph_list[i].edge_index[0].tolist()
                coo_dst = self.frag_graph_list[i].edge_index[1].tolist()
                coo_as_tupleset = {(min(s, d), max(s, d)) for s, d in zip(coo_src, coo_dst)}
                tuples_as_tupleset = {(min(a, b), max(a, b)) for a, b in wanted_frag_graphs[i]['edge_indices']}

                # Compare sets of edges
                self.assertEqual(coo_as_tupleset, tuples_as_tupleset, "Edge lists do not match for undirected graph")
    
    def test_jt_subgraph_correct_on_mol_with_unknown_atoms(self):
        uknown_at_smiles='O=[N+]([O-])c1ccc(Cl)c(Cl)c1Cl'
        mol = Chem.MolFromSmiles(uknown_at_smiles)
        graph = construct_dataset([uknown_at_smiles], None, graph_only=True)
        frag_graph_list, motif_graph, atom_mask, frag_flag = self.jt_subgraph.fragmentation(graph, mol)
        
        wanted_atom_mask = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
                                    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        wanted_frag_flag = ['ACN', 'ACCl', 'ACCl', 'ACCl', 'ACH', 'ACH', 'unknown', 'unknown']
        wanted_num_frag_graphs = 8
        wanted_motif_graph_num_nodes = 8
        wanted_motif_graph_num_edges = 16

        assert_array_equal(atom_mask[0], wanted_atom_mask[0]) #fails because of np error
        self.assertEqual(frag_flag, wanted_frag_flag)
        self.assertEqual(len(frag_graph_list), wanted_num_frag_graphs)
        self.assertEqual(motif_graph.num_nodes, wanted_motif_graph_num_nodes)
        print(motif_graph)
        print(motif_graph.edge_index)
        self.assertEqual(motif_graph.num_edges, wanted_motif_graph_num_edges)


if __name__ == '__main__':
    unittest.main()



from rdkit import Chem

# SMILES strings
# smiles1 = 'CC(=O)C1=CC=CC=C1'
# smiles2 = 'c1ccccc1(C(=O)C)'
# mol1 = Chem.CanonSmiles(smiles1)
# mol2 = Chem.CanonSmiles(smiles2)
