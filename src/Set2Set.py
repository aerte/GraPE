# a PyG implementation of Set-to-set
import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool

class Set2Set(nn.Module):
    def __init__(self, in_dim, device, num_iters=6, num_layers=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 2 * in_dim
        self.num_layers = num_layers
        self.num_iters = num_iters
        self.device = device
        self.lstm_output_dim = self.out_dim - self.in_dim
        self.lstm = nn.LSTM(self.out_dim, self.in_dim, num_layers=num_layers, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        #self.predict[0].reset_parameters()
        #other reset parameters if using more layers/components

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        h = (x.new_zeros((self.num_layers, batch_size, self.in_dim)),
             x.new_zeros((self.num_layers, batch_size, self.in_dim)))
        q_star = x.new_zeros(batch_size, self.out_dim)

        #here, we aren't storing the graph in a specific graph object,
        # we just have x (the data)
        for _ in range(self.num_iters):
            q, h = lstm(q_star.unsqueeze(1), h)
            q = q.view(batch_size, self.in_dim)
            e = torch.mul(x, q[batch]).sum(dim=-1)
            a = scatter(e, batch, dim=0, reduce='softmax')
            r = scatter(a.unsqueeze(-1) * x, batch, dim=0, reduce='sum')
            q_star = torch.cat([q, r], dim=-1)