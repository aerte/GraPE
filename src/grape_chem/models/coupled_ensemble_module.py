import torch
import torch.nn as nn
from grape_chem.models import GroupGAT

__all__ = ['GroupGAT_Ensemble']

class GroupGAT_Ensemble(nn.Module):
    """
    the base code for the "coupled single task"
    architecture
    """
    def __init__(self, net_params, num_targets, fn=None,):
        super().__init__()
        # self.models = nn.ModuleList([
        #     GroupGAT.GCGAT_v4pro_jit(net_params) for _ in range(num_targets)
        # ])
        # ^ If you want a model with variable number of coeffs, though it won't be jittable
        self.device = net_params['device']
        A_model = GroupGAT.GCGAT_v4pro(net_params)
        B_model = GroupGAT.GCGAT_v4pro(net_params)
        C_model = GroupGAT.GCGAT_v4pro(net_params)
        D_model = GroupGAT.GCGAT_v4pro(net_params)
        E_model = GroupGAT.GCGAT_v4pro(net_params)
        self.models = nn.ModuleDict({
            'A': A_model,
            'B': B_model,
            'C': C_model,
            'D': D_model,
            'E': E_model
        })

    def forward(self, data):
        T = data.global_feats.to(self.device)
        outputs = []
        A = self.models['A'](data)
        B = self.models['B'](data)
        C = self.models['C'](data)
        D = self.models['D'](data)
        E = self.models['E'](data)
        Cp = CpLayer()
        return Cp(A, B, C, D, E, T)
        
class CpLayer(nn.Module):
    def __init__(self):
        super(CpLayer, self).__init__()

    def forward(self, B, C, D, E, F, T):
        epsilon = 1e-7  # to avoid division by zero
        T = T.unsqueeze(1) + epsilon  # Ensure T has shape [700, 1]

        # Clamp D_over_T and F_over_T to prevent extreme values
        D_over_T = torch.clamp(D / T, min=-20, max=20)
        F_over_T = torch.clamp(F / T, min=-20, max=20)

        # Safe computations for sinh and cosh
        def safe_sinh(x):
            return torch.where(
                torch.abs(x) <= 20,
                torch.sinh(x),
                torch.sign(x) * (torch.exp(torch.clamp(torch.abs(x), max=88)) / 2)
            )

        def safe_cosh(x):
            return torch.where(
                torch.abs(x) <= 20,
                torch.cosh(x),
                torch.exp(torch.clamp(torch.abs(x), max=88)) / 2
            )

        sinh_term = safe_sinh(D_over_T) + epsilon
        cosh_term = safe_cosh(F_over_T) + epsilon

        # Debugging statements
        # print(f"D_over_T stats - min: {D_over_T.min()}, max: {D_over_T.max()}, mean: {D_over_T.mean()}")
        # print(f"sinh_term stats - min: {sinh_term.min()}, max: {sinh_term.max()}, mean: {sinh_term.mean()}")

        Cp = B + C * ((D_over_T / sinh_term) ** 2) + E * ((F_over_T / cosh_term) ** 2)
        return Cp