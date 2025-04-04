import os
import time
import math
from typing import Any, Dict, Optional, Tuple

import hydra
from hydra.utils import get_original_cwd
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchmetrics import MeanMetric

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image
from dem.utils.data_utils import remove_mean, calculate_rmsd_matrix

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.distribution_distances import compute_distribution_distances
from .components.ema import EMAWrapper
from .components.lambda_weighter import BaseLambdaWeighter
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt
from .components.score_scaler import BaseScoreScaler
from .components.sde_integration import integrate_sde
from .components.sdes import SDE

logtwopi = math.log(2 * math.pi)


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def t_stratified_loss(batch_t, batch_loss, num_bins=5, loss_name=None):
    """Stratify loss by binning t."""
    flat_losses = batch_loss.flatten().detach().cpu().numpy()
    flat_t = batch_t.flatten().detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


class DDSLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        tnet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        compile: bool,
        clipper: Optional[Clipper] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        dds_scale=1.0,
        time_range=1.0,
        use_ema=False,
        version=1
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param buffer: Buffer of sampled objects
        """
        super().__init__()
        # Seems to slow things down
        # torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.dim = self.energy_function.dimensionality
        self.time_range = time_range
        self.dds_scale = dds_scale
        self.diffusion_scale = diffusion_scale

        self.net = net(energy_function=energy_function)
        self.tcond = tnet()

        if use_ema:
            self.net = EMAWrapper(self.net)

        self.score_scaler = None

        self.num_samples_to_generate_per_epoch = num_samples_to_generate_per_epoch
        self.eval_batch_size = eval_batch_size
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_save = num_samples_to_save
        
        self.last_samples = None
        self.last_energies = None

        self.partial_prior = partial_prior
        self.clipper_gen = clipper_gen

        self.outputs = {}

        self.dds_train_loss = MeanMetric()
        self.dds_prior_ll = MeanMetric()
        self.dds_sample_ll = MeanMetric()
        self.dds_reg_loss = MeanMetric()
        self.dds_term_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_energy_w2 = MeanMetric()
        self.val_dist_total_var = MeanMetric()
        self.val_data_total_var = MeanMetric()
        self.val_dist_w2 = MeanMetric()
        self.val_data_w2 = MeanMetric()

    def score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.energy_function(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data
        
    def drift(self, t, x):
        if self.clipper_gen is not None:
            score_func = self.clipper_gen.wrap_grad_fxn(self.score)
        grad = score_func(x)
        f = torch.clip(self.net(t, x), -1e4, 1e4)
        dx = torch.nan_to_num(f + self.tcond(t) * grad)
        return dx

    def get_loss(self):
        prior_samples = self.prior.sample(self.eval_batch_size).to(self.device)

        x_1, r_k_reg = self.integrate(
            self.dds_sde,
            prior_samples,
            return_full_trajectory=False,
            no_grad=False,
            reverse_time=False,
            time_range=self.time_range,
        )
        
        prior_ll = self.prior.log_prob(x_1) 
        sample_ll = self.energy_function(x_1).sum(-1)
        term_loss = prior_ll - sample_ll
        dds_loss = (term_loss + r_k_reg).mean()

        return dds_loss, prior_ll, sample_ll, r_k_reg, term_loss 

    def generate_samples(
        self,
        sde,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale=1.0,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch
        samples = self.prior.sample(num_samples).to(self.device)
        return self.integrate(
            sde=sde,
            samples=samples,
            reverse_time=False,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
            no_grad=True,
            time_range=self.time_range
        )[0]

    def integrate(
        self,
        sde=None,
        samples: torch.Tensor = None,
        reverse_time=True,
        return_full_trajectory=False,
        diffusion_scale=1.0,
        no_grad=False,
        time_range=1.0,
    ) -> torch.Tensor:
        trajectory, r_k_reg = integrate_sde(
            sde or self.dds_sde,
            samples,
            self.num_integration_steps,
            self.energy_function,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
            time_range=time_range,
            var_preserve=True
        )
        if return_full_trajectory:
            return trajectory, r_k_reg
        return trajectory[-1], r_k_reg


    def training_step(self, batch, batch_idx):
        loss = 0.0
        dds_loss, prior_ll, sample_ll, quad_reg, term_loss = self.get_loss()
        self.dds_train_loss(dds_loss)
        self.dds_prior_ll(prior_ll)
        self.dds_sample_ll(sample_ll)
        self.dds_reg_loss(quad_reg)
        self.dds_term_loss(term_loss)

        loss = loss + dds_loss

        self.log_dict(
            {
                "train/dds_loss": self.dds_train_loss,
                "train/dds_prior_ll": self.dds_prior_ll,
                "train/dds_sample_ll": self.dds_sample_ll,
                "train/dds_reg_loss": self.dds_reg_loss,
                "train/dds_term_loss": self.dds_term_loss,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.hparams.use_ema:
            self.net.update_ema()
            if self.should_train_cfm(batch_idx):
                self.cfm_net.update_ema()

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # self.last_samples = self.generate_samples()
        # self.last_energies = self.energy_function(self.last_samples)
        self.last_samples = self.generate_samples(
            self.dds_sde, diffusion_scale=self.diffusion_scale
        )
        self.last_energies = self.energy_function(self.last_samples).sum(-1)
        
        prefix = 'val'
        self._log_energy_w2(prefix=prefix)
        self._log_data_w2(prefix=prefix)
        if self.energy_function.is_molecule:
            self._log_dist_total_var(prefix=prefix)
            self._log_dist_w2(prefix=prefix)
        else:
            self._log_data_total_var(prefix=prefix)
        sample_path = f"traj_data/dds_{self.energy_function.name}"
        os.makedirs(sample_path, exist_ok=True)
        torch.save(self.last_samples.detach().cpu().float(), os.path.join(sample_path, f"samples_{self.iter_num}.pt"))

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch = self.energy_function.sample_test_set(self.num_samples_to_generate_per_epoch)
        loss = self.get_loss()[0]

        # update and log metrics
        loss_metric = self.val_loss if prefix == "val" else self.test_loss
        loss_metric(loss)

        if self.last_samples is None:
            self.last_samples = self.generate_samples(
                self.dds_sde, diffusion_scale=self.diffusion_scale
            )

        self.outputs[f"{prefix}/data"] = batch
        self.outputs[f"{prefix}/gen"] = self.last_samples

        self.log(f"{prefix}/loss", loss_metric, on_step=False, on_epoch=True, prog_bar=True)

        batch = self.energy_function.sample_test_set(self.eval_batch_size)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)


    def scatter_prior(self, prefix, outputs):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        fig, ax = plt.subplots()
        n_samples = outputs.shape[0]
        ax.scatter(*outputs.detach().cpu().T, label="Generated prior")
        ax.scatter(
            *self.prior.sample(n_samples).cpu().T,
            label="True prior",
            alpha=0.5,
        )
        ax.legend()
        wandb_logger.log_image(f"{prefix}/generated_prior", [fig_to_image(fig)])

    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)

        buffer_samples = self.last_samples
        buffer_samples = self.energy_function.unnormalize(buffer_samples)
        self.energy_function.log_samples(
            buffer_samples, wandb_logger, f"{prefix}_samples/buffer_samples"
        )
        self.outputs = {}

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)
        
        self.eval_epoch_end("test")
        
        self._log_energy_w2(prefix="test")
        self._log_data_w2(prefix="test")
        self._log_energy_w2(prefix="test")
        self._log_data_w2(prefix="test")
        
        if self.energy_function.is_molecule:
            self._log_dist_total_var(prefix="test")
            self._log_dist_w2(prefix="test")
        else:
            self._log_data_total_var(prefix="test")
            
        batch_size = 1000
        final_samples = []
        n_batches = self.num_samples_to_save // batch_size
        print("Generating samples")
        for i in range(n_batches):
            start = time.time()
            samples = self.generate_samples(
                self.dds_sde, 
                diffusion_scale=self.diffusion_scale,
                num_samples=batch_size
            )
            final_samples.append(samples)
            end = time.time()
            print(f"batch {i} took {end - start:0.2f}s")

            if i == 0:
                self.energy_function.log_on_epoch_end(
                    samples,
                    self.energy_function(samples).sum(-1),
                    wandb_logger,
                )

        final_samples = torch.cat(final_samples, dim=0)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        path = f"{output_dir}/samples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")

        os.makedirs(self.energy_function.name, exist_ok=True)
        path2 = f"{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path2)
        print(f"Saving samples to {path2}")

    def _log_energy_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                self.dds_sde,
                num_samples=self.eval_batch_size, 
                diffusion_scale=self.diffusion_scale
            )
            generated_energies = self.energy_function(generated_samples).sum(-1)
            generated_samples = generated_samples[generated_energies > -100]
        else:
            if self.last_samples is None:
                return
            if len(self.last_samples) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, generated_energies = self.last_samples, self.last_energies
        energies = self.energy_function(self.energy_function.normalize(data_set)).sum(-1)
        energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())#is there something wrong here? weird large number

        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    def _log_data_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                self.dds_sde,
                num_samples=self.eval_batch_size, 
                diffusion_scale=self.diffusion_scale
            )

        else:
            if self.last_samples is None:
                return
            if len(self.last_samples) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, generated_energies = self.last_samples, self.last_energies
        generated_samples = self.energy_function.unnormalize(generated_samples)
        if self.energy_function.is_molecule:
            distance_matrix = calculate_rmsd_matrix(data_set.view(-1, 
                                                                  self.energy_function.n_particles,
                                                                  self.energy_function.n_spatial_dim),
                                                    generated_samples.view(-1, 
                                                                  self.energy_function.n_particles,
                                                                  self.energy_function.n_spatial_dim)).cpu().numpy()
        else:
            distance_matrix = pot.dist(data_set.cpu().numpy(), generated_samples.cpu().numpy(), metric='euclidean')
        src, dist = np.ones(len(data_set)) / len(data_set), np.ones(len(generated_samples)) / len(generated_samples)
        G = pot.emd(src, dist, distance_matrix)
        w2_dist = np.sum(G * distance_matrix) / G.sum()
        w2_dist = torch.tensor(w2_dist, device=data_set.device)
        self.log(
            f"{prefix}/data_w2",
            self.val_energy_w2(w2_dist),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    def _log_dist_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                self.dds_sde, 
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if self.last_samples is None:
                return
            if len(self.last_samples) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, generated_energies = self.last_samples, self.last_energies

        dist_w2 = pot.emd2_1d(
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )
        self.log(
            f"{prefix}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    def _log_dist_total_var(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                self.dds_sde,
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if self.last_samples is None:
                return
            if len(self.last_samples) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, generated_energies = self.last_samples, self.last_energies
 

        generated_samples_dists = (
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
        )
        data_set_dists = self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1)

        H_data_set, x_data_set = np.histogram(data_set_dists, bins=200)
        H_generated_samples, _ = np.histogram(generated_samples_dists, bins=(x_data_set))
        total_var = (
            0.5
            * np.abs(
                H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )

        self.log(
            f"{prefix}/dist_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    def _log_data_total_var(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                self.dds_sde,
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if self.last_samples is None:
                return
            if len(self.last_samples) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, generated_energies = self.last_samples, self.last_energies
        
        bins = (200, ) * self.energy_function.dimensionality
        generated_samples = self.energy_function.unnormalize(generated_samples)
        all_data = torch.cat([data_set, generated_samples], dim=0)
        min_vals, _ = all_data.min(dim=0)
        max_vals, _ = all_data.max(dim=0)
        ranges = tuple((min_vals[i].item(), max_vals[i].item()) for i in range(self.energy_function.dimensionality))  # tuple of (min, max) for each dimension
        ranges = tuple(item for subtuple in ranges for item in subtuple)
        hist_p, _ = torch.histogramdd(data_set.cpu(), bins=bins, range=ranges)
        hist_q, _ = torch.histogramdd(generated_samples.cpu(), bins=bins, range=ranges)
        
        p_dist = hist_p / hist_p.sum()
        q_dist = hist_q / hist_q.sum()
        
        total_var = 0.5 * torch.abs(p_dist - q_dist).sum()
        self.log(
            f"{prefix}/data_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        self.prior = self.partial_prior(
            device=self.device, scale=self.dds_scale * np.sqrt(self.time_range)
        )
        self.net = self.net.to(self.device)
        self.tcond = self.tcond.to(self.device)
        self.dds_sde = SDE(self.drift, None, 
                           noise_schedule=self.noise_schedule).to(self.device)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.lr_scheduler_update_frequency,
                },
            }
        return {"optimizer": optimizer}
