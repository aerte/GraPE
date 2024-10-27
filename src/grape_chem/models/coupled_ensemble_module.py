import torch
import torch.nn as nn
from grape_chem.models import GroupGAT_jittable

__all__ = ['GroupGAT_Ensemble']

class GroupGAT_Ensemble(nn.Module):
    """
    the base code for the "coupled single task"
    architecture
    """
    def __init__(self, net_params, num_targets, fn=None):
        super().__init__()
        self.models = nn.ModuleList([
            GroupGAT_jittable.GCGAT_v4pro_jit(net_params) for _ in range(num_targets)
        ])
        self.fn = fn

    def forward(self, data):
        outputs = []
        for model in self.models:
            output = model(data)
            outputs.append(output)
        if self.fn:
            return self.fn(outputs)
        else:
            return torch.stack(outputs, dim=1)