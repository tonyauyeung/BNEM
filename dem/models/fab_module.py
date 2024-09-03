import time
import copy
import math
from typing import Any, Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from hydra.utils import get_original_cwd
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchmetrics import MeanMetric

from fab.sampling_methods.transition_operators.base import TransitionOperator
from fab.sampling_methods.transition_operators.metropolis import Metropolis


from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.data_utils import remove_mean, calculate_rmsd_matrix
from dem.utils.logging_utils import fig_to_image

from .components.clipper import Clipper
from .components.ffjord import FFJORD
from .components.ema import EMAWrapper
from .components.ema import EMA

from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt
from .components.score_scaler import BaseScoreScaler
from .components.ais_fab import AnnealedImportanceSampler


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


class FABLitModule(LightningModule):
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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        use_ema=False,
        use_exact_likelihood=False,
        n_ais_intermediate_distributions=1,
        n_ais_inner_steps=5,
        ais_init_step_size=5.,
        init_from_prior=False,
        use_buffer=True,
        alpha=2,#parameter for alpha-divergence
        ema_beta=0.95,
        ema_steps=0,
        version=1,
        compile=False,
        
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
        self.cfm_net = net(energy_function=energy_function)

        self.EMA = EMA(beta=ema_beta, step_start_ema=ema_steps)
        self.ema_model = copy.deepcopy(self.cfm_net).eval().requires_grad_(False)
        
        if use_ema:
            self.cfm_net = EMAWrapper(self.cfm_net)
        if input_scaling_factor is not None or output_scaling_factor is not None:

            self.cfm_net = ScalingWrapper(
                self.cfm_net, input_scaling_factor, output_scaling_factor
            )

        self.score_scaler = None
        if score_scaler is not None:
            self.score_scaler = self.hparams.score_scaler(noise_schedule)
            self.cfm_net = self.score_scaler.wrap_model_for_unscaling(self.cfm_net)

        self.fab_cnf = FFJORD(
            self.cfm_net,
            trace_method='exact' if use_exact_likelihood else 'hutch',
            num_steps=num_integration_steps
        )


        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.buffer = buffer
        self.use_buffer = use_buffer
        self.dim = self.energy_function.dimensionality
            
        grad_fxn = estimate_grad_Rt

        self.clipper = clipper
        self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(grad_fxn)

        self.train_loss = MeanMetric()

        self.val_energy_w2 = MeanMetric()
        self.val_dist_total_var = MeanMetric()
        self.val_ddata_total_var = MeanMetric()
        self.val_dist_w2 = MeanMetric()
        self.val_data_w2 = MeanMetric()

        self.num_init_samples = num_init_samples
        self.num_samples_to_generate_per_epoch = num_samples_to_generate_per_epoch
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_save = num_samples_to_save
        self.eval_batch_size = eval_batch_size

        self.last_samples = None
        self.last_energies = None
        self.last_likelihood = None
        self.last_log_w = None
        self.eval_step_outputs = []

        self.partial_prior = partial_prior

        self.clipper_gen = clipper_gen

        self.init_from_prior = init_from_prior
        self.alpha = alpha
        
        ais_transition = Metropolis(n_ais_intermediate_distributions=n_ais_intermediate_distributions,
                 dim=self.energy_function.dimensionality,
                 base_log_prob=lambda x: - self.compute_nll(self.fab_cnf,
                                                      self.prior, x),
                 target_log_prob=self.energy_function,
                 n_updates=n_ais_inner_steps, 
                 max_step_size=ais_init_step_size,
                 alpha=alpha)
        
        self.ais_sampler = AnnealedImportanceSampler(self.energy_function, 
                                                     ais_transition,
                                                     p_target=False,
                                                     alpha=alpha,
                                                     n_intermediate_distributions=n_ais_intermediate_distributions
                                                     )
        
        
        self.iter_num = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.cfm_net(t, x)


    def get_loss(self, samples: torch.Tensor, samples_log_w: torch.Tensor, 
                 samples_log_q: torch.Tensor, indices) -> torch.Tensor:
        log_q = - self.compute_nll(self.fab_cnf, self.prior, samples)
        log_w_correct = (samples_log_q - log_q)  * (self.alpha - 1)
        
        #update the samples in the buffer
        self.buffer.buffer.log_q[indices] = log_q
        log_w_adjust = (log_w_correct + samples_log_w)
        self.buffer.buffer.log_w[indices] = log_w_adjust
        w_correct = torch.clamp(torch.exp(log_w_correct), max=1e3).detach()
        loss = - torch.mean(w_correct * log_q)
        return loss
        

        

    def training_step(self, batch, batch_idx):
        
        if self.use_buffer:
            iter_samples, log_w, log_q, indices = self.buffer.sample(self.num_samples_to_sample_from_buffer)
        else:
            iter_samples = self.prior.sample(self.num_samples_to_sample_from_buffer)
            
        if self.energy_function.is_molecule:
            iter_samples = remove_mean(
                iter_samples,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )

        fab_loss = self.get_loss(iter_samples, log_w, log_q, indices)

        # update and log metrics
        self.train_loss(fab_loss)
        self.log(
            "train/fab_loss",
            fab_loss.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return fab_loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.hparams.use_ema:
            self.cfm_net.update_ema()

    def generate_samples(
        self,
        cnf: FFJORD,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch

        samples = self.prior.sample(num_samples)
        self.EMA.step_ema(self.ema_model, self.cfm_net)
        with torch.no_grad():
            test_samples =  cnf.reverse_fn(samples)[-1]
        return test_samples
    
    
    def compute_nll(
        self,
        cnf,
        prior,
        samples: torch.Tensor,
    ):
        '''
        aug_samples = torch.cat(
            [samples, torch.zeros(samples.shape[0], 1, device=samples.device)], dim=-1
        )
        aug_output = cnf.integrate(aug_samples)[-1]
        x_1, logdetjac = aug_output[..., :-1], aug_output[..., -1]
        #if not cnf.is_diffusion:
        #    logdetjac = -logdetjac
        log_p_1 = prior.log_prob(x_1)
        log_p_0 = log_p_1 + logdetjac
        nll = -log_p_0
        '''
        z, delta_logp, reg_term = cnf.forward(samples)
        nll = - (prior.log_prob(z) + delta_logp.view(-1))
        return nll

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.ais_sampler.transition_operator.set_eval_mode(True)
        if self.clipper_gen is not None:
            reverse_sde = FFJORD(
                self.clipper_gen.wrap_grad_fxn(self.cfm_net),
            )
        else:
            reverse_sde = self.fab_cnf
        
        flow_samples = self.generate_samples(reverse_sde, num_samples=self.num_samples_to_generate_per_epoch)
        flow_likelihood = self.compute_nll(reverse_sde,
                                                self.prior,
                                                copy.deepcopy(flow_samples))
        ais_samples, log_w = self.ais_sampler.sample_and_log_weights(copy.deepcopy(flow_samples), 
                                                                     lambda x: - self.compute_nll(reverse_sde,
                                                                    self.prior, x),
                                                                     - flow_likelihood, False)
        ais_likelihood = - self.compute_nll(reverse_sde,
                                                self.prior,
                                                ais_samples)
        
        self.last_samples = flow_samples
        self.last_energies = self.energy_function(self.last_samples)
        
        self.buffer.add(ais_samples.detach(), log_w.detach(), ais_likelihood.detach())
        prefix = "val"

        
        self._log_energy_w2(prefix=prefix)
        self._log_data_w2(prefix=prefix)
        
        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix=prefix)
            self._log_dist_total_var(prefix=prefix)
        else:
            self._log_data_total_var(prefix=prefix)
        self.ais_sampler.transition_operator.set_eval_mode(False)

    def _log_energy_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                self.fab_cnf,
                num_samples=self.eval_batch_size, 
            )
            generated_energies = self.energy_function(generated_samples)
            generated_samples = generated_samples[generated_energies > -100]
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)
        energies = self.energy_function(self.energy_function.normalize(data_set))
        generated_energies = self.energy_function(generated_samples)
        energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())

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
                self.fab_cnf,
                num_samples=self.eval_batch_size
            )

        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)
        
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
                self.fab_cnf,
                num_samples=self.eval_batch_size
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

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
                self.fab_cnf,
                num_samples=self.eval_batch_size
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

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
                self.fab_cnf,
                num_samples=self.eval_batch_size
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)
        
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

    def _log_dist_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                self.fab_cnf,
                num_samples=self.eval_batch_size
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
                self.fab_cnf,
                num_samples=self.eval_batch_size
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
                self.fab_cnf,
                num_samples=self.eval_batch_size
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


    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if prefix == "test":
            batch = self.energy_function.sample_test_set(self.eval_batch_size)
        elif prefix == "val":
            batch = self.energy_function.sample_val_set(self.eval_batch_size)

        batch = self.energy_function.normalize(batch)
        backwards_samples = self.last_samples

        # generate samples noise --> data if needed
        if backwards_samples is None or self.eval_batch_size > len(backwards_samples):
            backwards_samples = self.generate_samples(
                self.fab_cnf,
                num_samples=self.eval_batch_size
            )

        # sample eval_batch_size from generated samples from dem to match dimensions
        # required for distribution metrics
        if len(backwards_samples) != self.eval_batch_size:
            indices = torch.randperm(len(backwards_samples))[: self.eval_batch_size]
            backwards_samples = backwards_samples[indices]

        if batch is None:
            print("Warning batch is None skipping eval")
            self.eval_step_outputs.append({"gen_0": backwards_samples})
            return

        times = torch.rand((self.eval_batch_size,), device=batch.device)

        noised_batch = batch + (
            torch.randn_like(batch) * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
        )

        if self.energy_function.is_molecule:
            noised_batch = remove_mean(
                noised_batch,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )

        # update and log metrics
        to_log = {
            "data_0": batch,
            "gen_0": backwards_samples,
        }
            
        self.eval_step_outputs.append(to_log)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)

    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)

        # Only plot dem samples
        self.energy_function.log_on_epoch_end(
            self.last_samples,
            self.last_energies,
            wandb_logger,
        )
        self.eval_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        self.eval_epoch_end("test")
        self._log_energy_w2(prefix="test")
        self._log_data_w2(prefix="test")
        if self.energy_function.is_molecule:
           self._log_dist_w2(prefix="test")
           self._log_dist_total_var(prefix="test")
        else:
            self._log_data_total_var(prefix="test")

        batch_size = 1000
        final_samples = []
        n_batches = self.num_samples_to_save // batch_size
        print("Generating samples")
        for i in range(n_batches):
            start = time.time()
            samples = self.generate_samples(
                self.fab_cnf,
                num_samples=batch_size
            )
            final_samples.append(samples)
            end = time.time()
            print(f"batch {i} took {end - start:0.2f}s")

            if i == 0:
                self.energy_function.log_on_epoch_end(
                    samples,
                    self.energy_function(samples),
                    wandb_logger,
                )

        final_samples = torch.cat(final_samples, dim=0)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        path = f"{output_dir}/samples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")
        import os

        os.makedirs(self.energy_function.name, exist_ok=True)
        path2 = f"{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path2)
        print(f"Saving samples to {path2}")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        self.prior = self.partial_prior(device=self.device, scale=self.noise_schedule.h(1) ** 0.5)
        
        if self.init_from_prior:
            flow_samples = self.prior.sample(self.num_init_samples)
        else:
            flow_samples = self.generate_samples(
                self.fab_cnf, self.num_init_samples, 
            )
        init_energies = self.energy_function(flow_samples)
        
        self.energy_function.log_on_epoch_end(
                flow_samples, init_energies,
                get_wandb_logger(self.loggers)
            )
        
        #compute nll using prior for init
        self.ais_sampler.transition_operator = self.ais_sampler.transition_operator.to(self.device)
        self.cfm_net = self.cfm_net.to(self.device)
        
        flow_likelihood = - self.compute_nll(self.fab_cnf,
                                          self.prior,
                                          flow_samples)
        ais_samples, log_w = self.ais_sampler.sample_and_log_weights(copy.deepcopy(flow_samples), 
                                                                     lambda x: - self.compute_nll(self.fab_cnf, self.prior, x),
                                                                     flow_likelihood, False)
        ais_likelihood = self.compute_nll(self.fab_cnf,
                                          self.prior,
                                          ais_samples)
        
        self.last_samples = flow_samples
        self.last_energies = self.energy_function(self.last_samples)
        
        
        self.buffer.add(ais_samples.detach(), log_w.detach(), ais_likelihood.detach())

        if self.hparams.compile and stage == "fit":
            self.cfm_net = torch.compile(self.cfm_net)


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
                    "monitor": "val/energy_w2",
                    "interval": "epoch",
                    "frequency": self.hparams.lr_scheduler_update_frequency,
                },
            }
        return {"optimizer": optimizer}
