import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class EnergyNet(nn.Module):
    def __init__(
        self, 
        net: nn.Module,
        energy_function: None,
    ):
        super(EnergyNet, self).__init__()
        self.is_molecule = energy_function.is_molecule
        if self.is_molecule:
            self.score_net = net(energy_function=energy_function, 
                             add_virtual=False)
        else:
            self.score_net = net(energy_function=energy_function)
        self.energy_function = energy_function
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward_e(self, t, y):
        score = self.score_net(t, y)
        if not self.energy_function.is_molecule:
            return - torch.linalg.vector_norm(score, dim=-1) + self.c
        else:
            score = score.view(score.shape[0], 
                               self.energy_function.n_particles,
                               self.energy_function.n_spatial_dim)
            return - torch.linalg.vector_norm(score, dim=-1).sum(-1) + self.c
    
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
    
        
