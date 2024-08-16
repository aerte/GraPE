import unittest
import torch
from torch_geometric.data import Data
#from your_module import remove_edges  # Replace with the actual module name where the function is defined

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
    new_data = Data(x=data.x, num_nodes=data.num_nodes, edge_index=new_edge_index)
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
            to_remove = [(0, 1)]
            updated_data = remove_edges(self.data, to_remove)

            expected_edge_index = torch.tensor([[2, 3],
                                                [3, 0]])
            expected_edge_attr = torch.tensor([[3], [4]])

            self.assertTrue(torch.equal(updated_data.edge_index, expected_edge_index))
            self.assertTrue(torch.equal(updated_data.edge_attr, expected_edge_attr))

    def test_remove_edges_only_acts_on_edges(self):
        to_remove = [(0, 1)]
        updated_data = remove_edges(self.data, to_remove)

        # Ensure node features are unchanged
        self.assertTrue(torch.equal(updated_data.x, self.data.x))

        # Ensure num_nodes is unchanged
        self.assertEqual(updated_data.num_nodes, self.data.num_nodes)

    def test_remove_edges_also_acts_on_edge_attr(self):
        to_remove = [(1, 2)]
        updated_data = remove_edges(self.data, to_remove)

        expected_edge_index = torch.tensor([[0, 3],
                                            [1, 0]])
        expected_edge_attr = torch.tensor([[1], [4]])

        self.assertTrue(torch.equal(updated_data.edge_index, expected_edge_index))
        self.assertTrue(torch.equal(updated_data.edge_attr, expected_edge_attr))

    def test_remove_edges_leaves_data_unchanged_when_no_edges_to_remove(self):
        to_remove = []
        updated_data = remove_edges(self.data, to_remove)

        self.assertTrue(torch.equal(updated_data.edge_index, self.data.edge_index))
        self.assertTrue(torch.equal(updated_data.edge_attr, self.data.edge_attr))
        self.assertTrue(torch.equal(updated_data.x, self.data.x))
        self.assertEqual(updated_data.num_nodes, self.data.num_nodes)

if __name__ == '__main__':
    unittest.main()