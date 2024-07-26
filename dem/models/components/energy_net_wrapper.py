import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class EnergyNet(nn.Module):
    def __init__(
        self, 
        net: nn.Module,
        energy_function: None,
        molecule_out_dim: int=32
    ):
        super(EnergyNet, self).__init__()
        self.is_molecule = energy_function.is_molecule
        if self.is_molecule:
            self.score_net = net(energy_function=energy_function, 
                             add_virtual=True)
        else:
            self.score_net = net(energy_function=energy_function)
        if not self.is_molecule:
            self.c = nn.Parameter(torch.tensor(0.0))
        self.energy_function = energy_function

    def forward_e(self, t, y):
        score = self.score_net(t, y)
        if not self.is_molecule:
            return score.sum(-1) + self.c
        else:
            return score.squeeze(-1)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, with_grad=False) -> torch.Tensor:
        """obtain score prediction of f_\theta(x, t) w.r.t. x"""
        x.requires_grad_(True)
        torch.set_grad_enabled(True)
        neg_energy = self.forward_e(t, x)
        score_x = torch.autograd.grad(neg_energy.sum(), x, create_graph=True, retain_graph=True)[0]
        if not with_grad:
            score_x = score_x.detach()
            x = x.detach()
            torch.set_grad_enabled(False)
        return score_x
    
        
