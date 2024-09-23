import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP
from typing import Union, List, Optional

__all__ = ['AFP_jittable']

class AFP_jittable(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers_atom: int = 3,
        num_layers_mol: int = 3,
        dropout: float = 0.0,
        regressor: bool = True,
        mlp_out_hidden: Union[int, List[int]] = 512,
        rep_dropout: float = 0.0,
        num_global_feats: int = 0,
        return_super_nodes: bool = False
    ):
        super(AFP_jittable, self).__init__()

        self.regressor = regressor
        self.num_global_feats = num_global_feats
        self.return_super_nodes = return_super_nodes

        self.rep_dropout = nn.Dropout(rep_dropout)

        if out_dim is None or out_dim == 1 or self.regressor:
            self.regressor = True
            if isinstance(mlp_out_hidden, int):
                out_dim = mlp_out_hidden
            else:
                out_dim = mlp_out_hidden[0]

        Attentive_FP = AttentiveFP  # Assuming this is TorchScript-compatible

        self.AFP_layers = Attentive_FP(
            in_channels=node_in_dim,
            edge_dim=edge_in_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=num_layers_atom,
            num_timesteps=num_layers_mol,
            dropout=dropout,
        )

        if self.regressor:
            if isinstance(mlp_out_hidden, int):
                self.mlp_out = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(out_dim + self.num_global_feats, mlp_out_hidden // 2),
                    nn.ReLU(),
                    nn.Linear(mlp_out_hidden // 2, 1)
                )
            else:
                layers = []
                for i in range(len(mlp_out_hidden)):
                    layers.append(nn.ReLU())
                    if i == len(mlp_out_hidden) - 1:
                        hidden_temp = mlp_out_hidden[i] + self.num_global_feats if i == 0 else mlp_out_hidden[i]
                        layers.append(nn.Linear(hidden_temp, 1))
                    elif i == 0:
                        layers.append(nn.Linear(mlp_out_hidden[i] + self.num_global_feats, mlp_out_hidden[i + 1]))
                    else:
                        layers.append(nn.Linear(mlp_out_hidden[i], mlp_out_hidden[i + 1]))
                self.mlp_out = nn.Sequential(*layers)
        else:
            self.mlp_out = nn.Identity()

    def reset_parameters(self):
        self.AFP_layers.reset_parameters()
        if self.regressor:
            for module in self.mlp_out.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        return_lats: bool,
        global_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.AFP_layers(x, edge_index, edge_attr, batch)
        if return_lats:
            return out
        # Dropout
        out = self.rep_dropout(out)

        if self.num_global_feats > 0 and global_feats is not None:
            out = torch.cat((out, global_feats), dim=1)
        if self.regressor:
            out = self.mlp_out(out)

        return out.view(-1)
