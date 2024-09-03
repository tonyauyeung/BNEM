from abc import ABC, abstractmethod
import math

import numpy as np
import torch


class BaseNoiseSchedule(ABC):
        
    @abstractmethod
    def g(t):
        # Returns g(t)
        pass

    @abstractmethod
    def h(t):
        # Returns \int_0^t g(t)^2 dt
        pass


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_max):
        self.beta = sigma_max

    def g(self, t):
        return torch.full_like(t, self.beta**0.5)

    def h(self, t):
        return self.beta * t
    
    def h_to_t(self, h):
        return h / self.beta
    
    def a(self, t, dt):
        t = torch.clamp(t, min=dt)
        a = 1 - math.exp(-2 * self.beta**0.5 * dt)
        return torch.zeros_like(t) + a


class QuadraticNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_max):
        self.beta = sigma_max

    def g(self, t):
        return torch.sqrt(self.beta * 2 * t)

    def h(self, t):
        return self.beta * t**2
    
    def h_to_t(self, h):
        return (h / self.beta + 1e-5).sqrt()
    
    def a(self, t, dt):
        t = torch.clamp(t, min=dt)
        return 1 - torch.exp(-2 * (2 * self.beta) ** 0.5 * (2 / 3) * (t ** 1.5 - (t - dt) ** 1.5))


class PowerNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_max, power):
        self.beta = sigma_max
        self.power = power

    def g(self, t):
        return torch.sqrt(self.beta * self.power * (t ** (self.power - 1)))

    def h(self, t):
        return self.beta * (t**self.power)
    
    def h_to_t(self, h):
        return h / self.beta ** (1 / self.power)
    
    def a(self, t, dt):
        t = torch.clamp(t, min=dt)
        p_tild = (self.power - 1) / 2 + 1
        return 1 - torch.exp(-2 * (self.beta / self.power) ** 0.5 / p_tild * (t ** p_tild - (t - dt) ** p_tild))


class GeometricNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = self.sigma_max / self.sigma_min

    def g(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then g(t) = sigma_min * sigma_d^t * sqrt{2 * log(sigma_d)}
        # See Eq 192 in https://arxiv.org/pdf/2206.00364.pdf
        return self.sigma_min * (self.sigma_diff**t) * ((2 * np.log(self.sigma_diff)) ** 0.5)

    def h(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then h(t) = \int_0^t g(z)^2 dz = sigma_min * sqrt{sigma_d^{2t} - 1}
        # see Eq 199 in https://arxiv.org/pdf/2206.00364.pdf
        return (self.sigma_min * (((self.sigma_diff ** (2 * t)) - 1) ** 0.5)) ** 2

    def h_to_t(self, h):
        return 0.5 * torch.log((h**0.5 / self.sigma_min) ** 2 + 1) / math.log(self.sigma_diff)
    
    def a(self, t, dt):
        t = torch.clamp(t, min=dt)
        return 1 - torch.exp(-2 * self.sigma_min * ((2 * np.log(self.sigma_diff)) ** 0.5) * ( 1 / math.log(self.sigma_diff)) * (torch.pow(torch.full_like(t, self.sigma_diff), t) - torch.pow(torch.full_like(t, self.sigma_diff), t - dt)))

class CosineNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_max, sigma_min=0.008):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        
    def g(self, t):
        tmp = np.pi / 2 * (1 - t + self.sigma_min) / (1 + self.sigma_min)
        if not isinstance(t, torch.Tensor):
            return math.sqrt(2 * np.pi / (1 + self.sigma_min) * math.sin(tmp) * math.cos(tmp)**3)
        else:
            return torch.sqrt(2 * np.pi / (1 + self.sigma_min) * torch.sin(tmp) * torch.cos(tmp).pow(3))

    def h(self, t):
        if not isinstance(t, torch.Tensor):
            return self.sigma_max * math.cos(np.pi / 2 * (1 - t + self.sigma_min) / (1 + self.sigma_min))**4
    
        else:
            return self.sigma_max * torch.cos(np.pi / 2 * (1 - t + self.sigma_min) / (1 + self.sigma_min)).pow(4)
    
class DDSNoiseSchedule(): #Discrete Cosine noise schedule
    def __init__(self, sigma_max, sigma_min=0.008):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def a(self, t, dt):
        return dt * torch.cos(np.pi / 2 * (1 - t + self.sigma_min) / (1 + self.sigma_min)).pow(4)
