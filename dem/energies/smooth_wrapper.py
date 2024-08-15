from abc import ABC, abstractmethod
from typing import Optional
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.data_utils import remove_mean

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

class SmoothEnergyFunction(ABC):
    def __init__(
        self,
        energy_func: BaseEnergyFunction,
        range_min: list,
        range_max: list,
        interpolation: 1000,
        threshold=1e8,
    ):
        self.energy_func = energy_func
        assert len(range_min) == len(range_max)
        self.ranges = list(zip(range_min, range_max))
        
        #fit spline cubic on these ranges
        interpolate_points = [torch.linspace(s_, e_, interpolation) for s_, e_ in self.ranges]
        
        es = [self.energy_func(x) for x in interpolate_points]
        for e, x in zip(interpolate_points, es):
            x = x[e < threshold]
            e = e[e < threshold]
        coeffs = [natural_cubic_spline_coeffs(x, e) for x, e in zip(interpolate_points, es)]
        self.splines = [NaturalCubicSpline(coeff) for coeff in coeffs]
        
    def smooth_call(self, samples: torch.Tensor) -> torch.Tensor:
        raw_energy = self.energy_func(samples)
        for i, ranges in enumerate(self.ranges):
            raw_energy[samples > ranges[0] & samples < ranges[1]] = self.splines[i](samples[samples > ranges[0] & samples < ranges[1]])
        return raw_energy