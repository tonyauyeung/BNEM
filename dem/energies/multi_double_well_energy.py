from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from hydra.utils import get_original_cwd
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.data_utils import remove_mean

from bgflow import Energy
from bgflow.utils import compute_distances



class MultiDoubleWellPotential(Energy):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via

    .. math::
        E_{DW}(d) = a \cdot (d-d_{\text{offset})^4 + b \cdot (d-d_{\text{offset})^2 + c.

    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of particles
    a, b, c, offset : float
        parameters of the potential
    """

    def __init__(self, dim, n_particles, a, b, c, offset, two_event_dims=True):
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._n_dimensions = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._n_dimensions)
        dists = dists - self._offset

        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        energies = torch.cat([energies, energies], dim=-1).view(energies.shape[0], self._n_particles, -1)
        return energies.sum(-1)
    

class MultiDoubleWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path,
        data_path_train=None,
        data_path_val=None,
        data_from_efm=True,  # if False, data from EFM
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        is_molecule=True,
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        self.data_from_efm = data_from_efm

        if data_from_efm:
            self.name = "DW4_EFM"
        else:
            self.name = "DW4_EACF"

        if self.data_from_efm:
            if data_path_train is None:
                raise ValueError("DW4 is from EFM. No train data path provided")
            if data_path_val is None:
                raise ValueError("DW4 is from EFM. No val data path provided")

        # self.data_path = get_original_cwd() + "/" + data_path
        # self.data_path_train = get_original_cwd() + "/" + data_path_train
        # self.data_path_val = get_original_cwd() + "/" + data_path_val

        self.data_path = data_path
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val

        self.device = device

        self.val_set_size = 1000
        self.test_set_size = 1000
        self.train_set_size = 100000

        self.multi_double_well = MultiDoubleWellPotential(
            dim=dimensionality,
            n_particles=n_particles,
            a=0.9,
            b=-4,
            c=0,
            offset=4,
            two_event_dims=False,
        )

        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

    def __call__(self, samples: torch.Tensor, smooth=None) -> torch.Tensor:
        if len(samples.shape) >= 2:
            samples_shape = list(samples.shape[:-1])
            samples = samples.view(-1, samples.shape[-1])
            energy =  -self.multi_double_well.energy(samples)
            return energy.view(*samples_shape, self.n_particles)
        else:
            return - self.multi_double_well.energy(samples).squeeze(-1)
    
    def load_data(self, path, size=None):
        if path[-3:] == "npy":
            samples = np.load(path, allow_pickle=True)
            if size is not None:
                samples = samples[0][-size: ]
        else:
            samples = torch.load(path)
        return samples

    def setup_test_set(self):
        if self.data_from_efm:
            data = self.load_data(self.data_path)

        else:
            data = self.load_data(self.data_path, size=self.test_set_size)

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )

        return data

    def setup_train_set(self):
        if self.data_from_efm:
            data = self.load_data(self.data_path_train)
        else:
            data = self.load_data(self.data_path, size=-self.train_set_size)

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )

        return data

    def setup_val_set(self):
        if self.data_from_efm:
            data = self.load_data(self.data_path_val)
        else:
            data = self.load_data(self.data_path, size=self.test_set_size)

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )
        return data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.multi_double_well.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        samples = self.unnormalize(samples)
        samples_fig = self.get_dataset_fig(samples)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if latest_samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            samples_fig = self.get_dataset_fig(latest_samples)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if unprioritized_buffer_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples)

                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

        self.curr_epoch += 1

    def get_dataset_fig(self, samples):
        test_data_smaller = self.sample_test_set(1000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples).detach().cpu()
        dist_test = self.interatomic_dist(test_data_smaller).detach().cpu()

        axs[0].hist(
            dist_samples.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["generated data", "test data"])

        energy_samples = -self(samples).detach().detach().cpu().sum(-1)
        energy_test = -self(test_data_smaller).detach().detach().cpu().sum(-1)

        min_energy = -26
        max_energy = 0

        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
