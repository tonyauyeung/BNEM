import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class EnergyNet(nn.Module):
    def __init__(
        self, 
        net: nn.Module,
        max_iter: 100,
        energy_function: None,
        noise_schedule: None
    ):
        super(EnergyNet, self).__init__()
        self.is_molecule = energy_function.is_molecule
        if self.is_molecule:
            self.score_net = net(energy_function=energy_function, 
                             add_virtual=False,
                             energy=True)
        else:
            self.score_net = net(energy_function=energy_function)
        self.energy_function = energy_function
        self.c = nn.Parameter(torch.tensor(0.0))
        self.noise_schedule = noise_schedule
        self.max_iter = max_iter
        self.score_clipper = None

    def forward_e(self, t, y):
        score = self.score_net(t, y)
        if not self.energy_function.is_molecule:
            return - torch.linalg.vector_norm(score, dim=-1) + score.sum(-1) + self.c
        else:
            score, potential = score
            score = score.view(score.shape[0], 
                               self.energy_function.n_particles,
                               self.energy_function.n_spatial_dim)
            potential = potential.view(score.shape[0], 
                               self.energy_function.n_particles, -1)
            return - torch.linalg.vector_norm(score, dim=-1).sum(-1) + potential.sum(-1).sum(-1)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, 
                with_grad=False, return_E=False) -> torch.Tensor:
        """obtain score prediction of f_\theta(x, t) w.r.t. x"""
        x.requires_grad_(True)
        torch.set_grad_enabled(True)
        neg_energy = self.forward_e(t, x)
        score_x = torch.autograd.grad(neg_energy.sum(), x, create_graph=True, retain_graph=True)[0]
        
        if self.score_clipper is not None:
            score_x = self.score_clipper.clip_scores(score_x)
        
        if not with_grad:
            score_x = score_x.detach()
            x = x.detach()
            torch.set_grad_enabled(False)
        if not return_E:
            return score_x
        else:
            return score_x, neg_energy
    
    @torch.no_grad()
    def MH_sample(self, t: torch.Tensor, x: torch.Tensor,
                  dt, diffusion_scale=1.) -> torch.Tensor:
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)
        g = self.noise_schedule.g(t.unsqueeze(-1)) ** 2
        g_dt = self.noise_schedule.g((t - dt).unsqueeze(-1)) ** 2
        score_pred, neg_energy = self.forward(t, x, with_grad=False, return_E=True)
        
        if score_pred.std() < neg_energy.std():
        
            accept_flag = torch.ones(x.shape[0]).bool()
            x_neg = x.clone()
            for _ in range(self.max_iter):
                noise = torch.randn_like(x)
                x_prop = x + score_pred * g * dt + \
                    g.sqrt() * noise * math.sqrt(dt) * diffusion_scale
                energy_prop = self.forward_e(t - dt, x_prop)
                
                q_state = torch.exp(- torch.linalg.vector_norm(x_prop - x, dim=-1)/ \
                    (g * dt).squeeze(-1))
                q_prop = torch.exp(- torch.linalg.vector_norm(noise, dim=-1) / \
                        (g_dt * dt).squeeze(-1))
                accept_prob = torch.exp(neg_energy - energy_prop) * q_state / q_prop
                accept_prob = torch.clamp(accept_prob, 0, 1)
                accept_mask = torch.rand_like(accept_prob) > accept_prob
                x_neg = multi_index_torch(x_neg, accept_flag, accept_mask, x_prop[accept_mask])
                
                x, score_pred, neg_energy = x[~accept_mask], score_pred[~accept_mask], neg_energy[~accept_mask]
                g, g_dt, t = g[~accept_mask], g[~accept_mask], t[~accept_mask]
                accept_flag = multi_index_torch(accept_flag, accept_flag, accept_mask, False)
                
                if x.shape[0] == 0:
                    break
                
            noise = torch.randn_like(x)
            x_neg[accept_flag] = x + score_pred * g * dt + \
                g.sqrt() * noise * math.sqrt(dt) * diffusion_scale
            
        else:
            x_neg = x + score_pred * g * dt + \
                g.sqrt() * torch.randn_like(x) * math.sqrt(dt) * diffusion_scale
        return x_neg
    

def multi_index_torch(source, ind_1, ind_2, target):
    #source[ind1][ind2] = target
    tmp = source[ind_1].clone()
    tmp[ind_2] = target
    source[ind_1.clone()] = tmp
    return source
            
        
        
