import torch
from torch_geometric.data import Data
import Set2Set  # Assuming the Set2Set class is saved in set2set_module.py

def main():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], 
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    x = torch.randn(4, 5)  # Node features (4 nodes, 5 features each)

    # Create a batch vector (all nodes belong to the same graph in this example)
    batch = torch.zeros(4, dtype=torch.long)  
    data = Data(x=x, edge_index=edge_index)

    in_dim = 5  # Same as node feature dimension

    set2set = Set2Set(in_dim=in_dim)
    out = set2set(data.x, batch)

    print("G_out:")
    print(out)

if __name__ == "__main__":
    main()