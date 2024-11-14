from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from hydra.utils import get_original_cwd
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image

from fab.target_distributions import many_well
from fab.utils.plotting import plot_contours, plot_marginal_pair


def get_target_log_prob_marginal_pair(log_prob, i: int, j: int, total_dim: int):
    def log_prob_marginal_pair(x_2d):
        x = torch.zeros((x_2d.shape[0], total_dim))
        x[:, i] = x_2d[:, 0]
        x[:, j] = x_2d[:, 1]
        return log_prob(x)
    return log_prob_marginal_pair


class ManyWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        is_molecule=False,
    ):
        use_gpu = device != "cpu"
        self.many_well_energy = many_well.ManyWellEnergy(
            dim=dimensionality,
            use_gpu=use_gpu,
            normalised=False,
            a=-0.5, 
            b=-6.0, 
            c=1.
        )
        self.n_particles = 1
        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        assert data_normalization_factor == 1.0
        self.data_normalization_factor = data_normalization_factor

        self.device = device

        self.val_set_size = 1000
        self.test_set_size = 1000
        self.train_set_size = 100000
        self.true_samples = self.many_well_energy.sample((1000,))
        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

    def setup_test_set(self):
        return self.many_well_energy.sample((self.test_set_size,))

    def setup_train_set(self):
        return None

    def setup_val_set(self):
        return self.many_well_energy.sample((self.val_set_size,))

    def __call__(self, samples: torch.Tensor, smooth=None) -> torch.Tensor:
        data_shape = samples.shape
        assert len(data_shape) <= 3
        if len(data_shape) == 3:
            samples = samples.view(data_shape[0] * data_shape[1], data_shape[2])
        return self.many_well_energy.log_prob(samples).view(*data_shape[:-1]).unsqueeze(-1)

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
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            samples_fig = self.get_dataset_fig(latest_samples, true_samples=self.true_samples)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if unprioritized_buffer_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples, true_samples=self.true_samples)

                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        if wandb_logger is None:
            return
        
        samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_single_dataset_fig(self, samples, name):
        self.many_well_energy.to("cpu")
        plotting_bounds = (-3, 3)
        dim = self._dimensionality
        n_rows = dim // 2
        samples_modes = self.many_well_energy._test_set_modes
        fig, axs = plt.subplots(dim // 2, 2, sharex=True, sharey=True, figsize=(10, n_rows * 3))

        for i in range(n_rows):
            plot_contours(self.many_well_energy.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 0])
            plot_contours(self.many_well_energy.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 1])

            # plot flow samples
            plot_marginal_pair(samples, ax=axs[i, 0], bounds=plotting_bounds,
                            marginal_dims=(i * 2, i * 2 + 1))
            plot_marginal_pair(samples_modes, ax=axs[i, 1], bounds=plotting_bounds,
                            marginal_dims=(i * 2, i * 2 + 1))
            axs[i, 0].set_xlabel(f"dim {i * 2}")
            axs[i, 0].set_ylabel(f"dim {i * 2 + 1}")

            plt.tight_layout()
        axs[0, 0].set_title("samples")
        axs[0, 1].set_title("true modes")
        fig.suptitle(f"{name}")
        self.many_well_energy.to(self.device)

        return fig_to_image(fig)

    def get_dataset_fig(self, samples, true_samples=None):
        self.many_well_energy.to("cpu")
        plotting_bounds = (-3, 3)
        dim = self._dimensionality
        n_rows = dim // 2
        if true_samples is None:
            true_samples = self.many_well_energy.sample((samples.shape[0], ))

        if dim == 8:
            fig, axs = plt.subplots(dim // 2, 2, sharex=True, sharey=True, figsize=(10, n_rows * 3))
            for i in range(n_rows):
                plot_contours(self.many_well_energy.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 0])
                plot_contours(self.many_well_energy.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 1])

                # plot flow samples
                if samples:
                    plot_marginal_pair(samples, ax=axs[i, 0], bounds=plotting_bounds,
                                    marginal_dims=(i * 2, i * 2 + 1))
                    plot_marginal_pair(true_samples, ax=axs[i, 1], bounds=plotting_bounds,
                                    marginal_dims=(i * 2, i * 2 + 1))
                axs[i, 0].set_xlabel(f"dim {i * 2}")
                axs[i, 0].set_ylabel(f"dim {i * 2 + 1}")

                plt.tight_layout()
            axs[0, 0].set_title("generated samples")
            axs[0, 1].set_title("true samples")
        elif dim == 32:
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 1 * 3))
            log_prob_target = get_target_log_prob_marginal_pair(
                self.many_well_energy.log_prob, 0, 2, dim)
            plot_contours(log_prob_target, bounds=plotting_bounds, ax=axs[0])
            plot_contours(log_prob_target, bounds=plotting_bounds, ax=axs[1])

            # plot flow samples
            if samples is not None:
                plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds,
                                marginal_dims=(0, 2))
                plot_marginal_pair(true_samples, ax=axs[1], bounds=plotting_bounds,
                                marginal_dims=(0, 2))
            axs[0].set_xlabel(f"dim 1")
            axs[0].set_ylabel(f"dim 3")

            plt.tight_layout()
            axs[0].set_title("generated samples")
            axs[1].set_title("true samples")
        self.many_well_energy.to(self.device)

        return fig_to_image(fig)