import unittest
import torch
from torch_geometric.data import Data
#from your_module import remove_edges  # Replace with the actual module name where the function is defined


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

if __name__ == '__main__':
    unittest.main()