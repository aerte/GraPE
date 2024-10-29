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
        """
        Computes Cp using the provided equation:
        Cp = B + C * ((D / T) / sinh(D / T)) ** 2 + E * ((F / T) / cosh(F / T)) ** 2
        """
        epsilon = 1e-7 # to avoid 0 division
        T = T + epsilon

        D_over_T = torch.clamp(D / T, min=-1000, max=1000)
        F_over_T = torch.clamp(F / T, min=-1000, max=1000)

        # print(f"D_over_T stats - min: {D_over_T.min()}, max: {D_over_T.max()}, mean: {D_over_T.mean()}")
        # print(f"F_over_T stats - min: {F_over_T.min()}, max: {F_over_T.max()}, mean: {F_over_T.mean()}")

        sinh_term = torch.sinh(D_over_T)
        cosh_term = torch.cosh(F_over_T)
        # print(f"sinh_term stats - min: {sinh_term.min()}, max: {sinh_term.max()}, mean: {sinh_term.mean()}")
        # print(f"cosh_term stats - min: {cosh_term.min()}, max: {cosh_term.max()}, mean: {cosh_term.mean()}")

        Cp = B + C * ((D_over_T / sinh_term) ** 2) + E * ((F_over_T / cosh_term) ** 2)
        # Ensure T doesn't contain zero
        
        # Cp_o = B + C * ((D / T) / torch.sinh(D / T)) ** 2 + E * (
        #         (F / T) / torch.cosh(F / T)) ** 2
        return Cp