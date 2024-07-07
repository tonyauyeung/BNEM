import torch
import numpy as np
from dem.models.components.noise_schedules import *

class BootstrapSchedule:
    """
    The BootstrapSchedule is iterated during a Bootstrapping-sampler traing, following the pseudo-code below:
            bootstrap_scheduler = BootstrapSchedule(num_steps)
            for t_last, t in bootstrap_scheduler:
                x_t ~ q(x_t | x_0)
                predictor = net(x_t, t)
                bootstrap_estimator = f(x_t, t_last, net)
                loss = loss_fn(predictor, bootstrap_estimator)
    """
    def __init__(
        self,
    ) -> None:
        self.time_splits = self.time_spliter()
        self.time_splits = torch.cat((torch.tensor([0.]), self.time_splits))  # pad a 0 at the beginning
        self.index = 1
    
    def time_spliter(self) -> torch.Tensor:
        """
        split [0, 1] into num_bootstrap_steps sub-intervals
        return the split points (include 0 and 1), torch.Tensor of shape (num_bootstrap_steps + 1, )
        """
        raise NotImplementedError

    def __next__(self):
        if self.index < len(self.time_splits):
            res = self.time_splits[:self.index + 1]
            self.index += 1
            return res
        else:
            raise StopIteration
        
    def t_to_index(self, t: torch.Tensor) -> torch.Tensor:
        indexes = []
        for t_ in t:
            indexes.append(torch.sum(self.time_splits < t_).item() - 1)
        return torch.tensor(indexes, dtype=torch.long).to(t.device)  
    
    def index_to_t(self, index: torch.Tensor) -> torch.Tensor:
        return torch.concat([self.time_splits[index], self.time_splits[index + 1]]) # [t_last, t_current]
    
    def sample_t(self, index: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(index.float()) * (self.time_splits[index + 1] - self.time_splits[index]) + self.time_splits[index]
    
    
    def __iter__(self):
        self.index = 1
        return self
    
    def __len__(self):
        return len(self.time_splits) - 1


class LinearBootstrapSchedule(BootstrapSchedule):
    def __init__(
        self, 
        noise_scheduler: GeometricNoiseSchedule,#place holder
        num_bootstrap_steps: int
    ) -> None:
        self.num_bootstrap_steps = num_bootstrap_steps
        super().__init__()

    def time_spliter(self) -> torch.Tensor:
        return torch.linspace(0, 1, self.num_bootstrap_steps + 1)


class GeometricBootstrapSchedule(BootstrapSchedule):
    def __init__(
        self,
        noise_scheduler: GeometricNoiseSchedule,
        variance: float
    ) -> None:
        self.sigma_max = noise_scheduler.sigma_max
        self.sigma_min = noise_scheduler.sigma_min
        self.d = noise_scheduler.sigma_diff
        self.var = variance
        super().__init__()

    def time_spliter(self) -> torch.Tensor:
        k = round((self.sigma_max ** 2 - 2 * self.sigma_min ** 2) / self.var)
        t = [0.]
        for i in range(1, k + 1):
            t.append(min(np.log((i * self.var + self.sigma_min ** 2) / self.sigma_min ** 2 + 1) * 0.5 / np.log(self.d), 1.0))
        return torch.tensor(t)
    
