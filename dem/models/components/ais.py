import numpy as np
import torch



def hmc(samples, log_prob, score_func=None, step_size=0.1, num_steps=1, mass=1):
    current_samples = samples.clone()
    momentum = torch.randn_like(samples) * mass
    # TODO: if score_func is None, then set it as the gradient of log_prob, i.e. torch.autograd.grad(log_prob(x).sum(), x)[0]
    accept_rates = []
    for _ in range(num_steps):
        momentum += 0.5 * step_size * score_func(current_samples)
        new_samples = current_samples.clone() + step_size * momentum
        alpha = torch.clamp(torch.exp(log_prob(new_samples) - log_prob(current_samples)), max=1)
        mask = torch.rand_like(alpha) < alpha
        current_samples = torch.where(mask.unsqueeze(-1), new_samples, current_samples)
        accept_rates.append(mask.detach().sum() / np.prod(mask.shape))
    return current_samples

def ais(xt, t, num_samples, L, 
        noise_schedule, energy_func,
        dt=0.1,  mode='score',
        reduction=True):
    device = xt.device
    sigmas = noise_schedule.h(t)[:, None].to(device).sqrt()
    data_shape = list(xt.shape)[1:]
    noise = torch.randn(xt.shape[0], num_samples, *data_shape, device=device)
    
    torch.set_grad_enabled(True)
    xt.requires_grad_(True)
    xk = noise * sigmas.unsqueeze(-1) + xt.unsqueeze(1)
    logw = torch.zeros((xt.shape[0], num_samples)).to(device)
    """compute log(w) for numerical stability"""
    for k in range(1, L + 1):
        """1-step HMC"""
        logw += energy_func(xk).to(device) / L
        log_prob_k = lambda x: k / L * (energy_func(x)) - 0.5 * torch.sum((x - xt.unsqueeze(1)) ** 2, dim=-1) / sigmas ** 2
        score_k = lambda x: k / L * energy_func.score(x) - (x - xt.unsqueeze(1)) / sigmas.unsqueeze(-1) ** 2
        xk = hmc(xk, log_prob_k, score_k, step_size=dt, num_steps=1, mass=1)
        # logw += -gmm(xk) / L
    # xt.requires_grad_(True)
    if mode == 'score':
        energy = torch.logsumexp(logw, dim=1) - np.log(num_samples)
        score = torch.autograd.grad(-energy.sum(), 
                                    inputs=xt)[0]
        xt.requires_grad_(False)
        return score
    elif mode == 'energy':
        if reduction:
            energy = torch.logsumexp(logw, dim=1) - np.log(num_samples)
            xt.requires_grad_(False)
            return energy
        else:
            xt.requires_grad_(False)
            return -logw