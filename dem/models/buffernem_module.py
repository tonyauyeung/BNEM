import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb

from .endem_module import *
from dem.models.components.prioritised_replay_buffer import SimpleBuffer, SimpleBufferVerbose, SimpleReplayDataVerbose
from dem.models.components.sampler.sampler import LangevinDynamics
from dem.models.components.sampler.dyn_mcmc_warp import DynSamplerWrapper


def compute_gaussian_density(x, y, sigma2):
    """
    Compute the density of Gaussian N(y[i]; x[j], sigma^2[i]I) for each pair.
    
    Args:
        x (torch.Tensor): Tensor of shape (N, dim).
        y (torch.Tensor): Tensor of shape (M, dim).
        sigma2 (torch.Tensor): Tensor of shape (M,).
    
    Returns:
        torch.Tensor: Tensor of shape (M, N) containing densities.
    """
    N, dim = x.shape
    M, _ = y.shape

    # Compute pairwise squared distances (M, N)
    diff = y[:, None, :] - x[None, :, :]  # Shape: (M, N, dim)
    sq_dist = torch.sum(diff**2, dim=-1)  # Shape: (M, N)

    # Compute Gaussian densities
    coeff = (2 * torch.pi * sigma2).pow(-dim / 2)  # Shape: (M,)
    exp_term = torch.exp(-sq_dist / (2 * sigma2[:, None]))  # Shape: (M, N)
    
    densities = coeff[:, None] * exp_term  # Shape: (M, N)
    return densities


class BUFFERNEMLitModule(ENDEMLitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_estimator_mc_samples: int,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        nll_with_dem: bool,
        nll_on_buffer: bool,
        logz_with_cfm: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        use_otcfm: bool,
        nll_integration_method: str,
        use_richardsons: bool,
        compile: bool,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        use_ema=False,
        use_exact_likelihood=False,
        debug_use_train_data=False,
        init_from_prior=False,
        compute_nll_on_train_data=False,
        use_buffer=True,
        tol=1e-5,
        version=1,
        negative_time=False,
        num_negative_time_steps=100,
        ais_steps: int = 0,
        ais_dt: float = 0.1,
        ais_warmup: int = 100,
        ema_beta=0.99,
        t0_regulizer_weight=0.,
        bootstrap_schedule: BootstrapSchedule = None,
        bootstrap_warmup: int = 2e3,
        bootstrap_mc_samples: int = 80,
        epsilon_train=1e-4,
        prioritize_warmup=0,
        iden_t=True,
        mh_iter=0,
        num_efficient_samples=0,
        bootstrap_from_checkpoint=True,
    ) -> None:
            super().__init__(
                net=net,
                optimizer=optimizer,
                scheduler=scheduler,
                energy_function=energy_function,
                noise_schedule=noise_schedule,
                lambda_weighter=lambda_weighter,
                buffer=buffer,
                num_init_samples=num_init_samples,
                num_estimator_mc_samples=num_estimator_mc_samples,
                num_samples_to_generate_per_epoch=num_samples_to_generate_per_epoch,
                num_samples_to_sample_from_buffer=num_samples_to_sample_from_buffer,
                num_samples_to_save=num_samples_to_save,
                eval_batch_size=eval_batch_size,
                num_integration_steps=num_integration_steps,
                lr_scheduler_update_frequency=lr_scheduler_update_frequency,
                nll_with_cfm=nll_with_cfm,
                nll_with_dem=nll_with_dem,
                nll_on_buffer=nll_on_buffer,
                logz_with_cfm=logz_with_cfm,
                cfm_sigma=cfm_sigma,
                cfm_prior_std=cfm_prior_std,
                use_otcfm=use_otcfm,
                nll_integration_method=nll_integration_method,
                use_richardsons=use_richardsons,
                compile=compile,
                prioritize_cfm_training_samples=prioritize_cfm_training_samples,
                input_scaling_factor=input_scaling_factor,
                output_scaling_factor=output_scaling_factor,
                clipper=clipper,
                score_scaler=score_scaler,
                partial_prior=partial_prior,
                clipper_gen=clipper_gen,
                diffusion_scale=diffusion_scale,
                cfm_loss_weight=cfm_loss_weight,
                use_ema=use_ema,
                ema_beta=ema_beta,
                use_exact_likelihood=use_exact_likelihood,
                debug_use_train_data=debug_use_train_data,
                init_from_prior=init_from_prior,
                compute_nll_on_train_data=compute_nll_on_train_data,
                use_buffer=use_buffer,
                tol=tol,
                version=version,
                negative_time=negative_time,
                num_negative_time_steps=num_negative_time_steps,
                ais_steps=ais_steps,
                ais_dt=ais_dt,
                ais_warmup=ais_warmup,
                t0_regulizer_weight=t0_regulizer_weight,
                bootstrap_schedule=bootstrap_schedule,
                bootstrap_warmup=bootstrap_warmup,
                bootstrap_mc_samples=bootstrap_mc_samples,
                epsilon_train=epsilon_train,
                prioritize_warmup=prioritize_warmup,
                mh_iter=mh_iter,
                num_efficient_samples=num_efficient_samples,
                bootstrap_from_checkpoint=bootstrap_from_checkpoint,
            )
            self.mcmc_step_size = 0.0001
            self.mcmc_finetune_steps = 10

    def buffer_energy_estimator(self, xt, t, reduction=False):
        sigmas2 = self.noise_schedule.h(t)
        data_shape = list(xt.shape)[1:]
        if isinstance(self.buffer, SimpleBuffer) or isinstance(self.buffer, SimpleBufferVerbose):
            buffer_energy = self.buffer.buffer.energy
            buffer_samples = self.buffer.buffer.x
        else:
            raise NotImplementedError
        log_weights = -0.5 * ((xt[:, None, :] - buffer_samples[None, :, :]) ** 2).sum(dim=-1, keepdim=True) / sigmas2.unsqueeze(-1).unsqueeze(-1)
        # log_weights = compute_gaussian_density(buffer_samples, xt, sigmas2).unsqueeze(-1).log()
        log_weights[torch.where(log_weights == -torch.inf)] = -1e9
        energy_est = torch.logsumexp(buffer_energy.clone().detach() + log_weights, dim=1)
        # energy_est = buffer_energy.unsqueeze(0) * weights
        # if reduction:
        #     energy_est = torch.logsumexp(energy_est, dim=1) -\
        #         torch.log(torch.tensor(buffer_samples.shape[0])).to(xt.device)
        #     return energy_est
        return energy_est
    

    def get_loss(self, times: torch.Tensor, 
                 samples: torch.Tensor, 
                 clean_samples: torch.Tensor,
                 train=False) -> torch.Tensor:

        energy_est = self.buffer_energy_estimator(samples, times).detach()
        predicted_energy = self.net.forward_e(times, samples)
        
        # TODO: check
        energy_est = self.sum_energy_estimator(energy_est, self.num_estimator_mc_samples)
        energy_error_norm = torch.abs(predicted_energy - energy_est).pow(2)

        if not self.buffer.prioritize:
            self.log(
                    "energy_loss",
                    energy_error_norm.mean(),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )
        else:
            self.log(
                    "energy_loss",
                    torch.tensor(1e8).to(energy_error_norm.device),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )
        self.iter_num += 1
        full_loss = energy_error_norm.sum(-1) * self.lambda_weighter(times)
        return full_loss
    
    def on_train_epoch_start(self):
        # lg = LangevinDynamics(x=self.buffer.buffer.x,
        #                         energy_func=lambda x: -self.energy_function(x).sum(-1),
        #                         step_size=self.mcmc_step_size,
        #                         mh=True,
        #                         device=self.buffer.device)
        # self.mcmc_sampler = DynSamplerWrapper(lg, per_temp=False, target_acceptance_rate=0.6, alpha=0.25)
        # for _ in range(self.mcmc_finetune_steps):
        #     new_samples, acc, _ = self.mcmc_sampler.sample()
        #     # self.log(
        #     #     "train/mcmc_acc_rate",
        #     #     acc,
        #     #     on_step=True,
        #     #     on_epoch=False,
        #     #     prog_bar=False,
        #     # )
        # new_samples = new_samples.clone().detach()
        # self.buffer.add(new_samples, self.energy_function(new_samples))
        # self.mcmc_step_size = self.mcmc_sampler.sampler.step_size
        self.train_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.EMA.step_ema(self.ema_model, self.net)
        "Lightning hook that is called when a training epoch ends."
        self.log(
            "val/training_time",
            time.time() - self.train_start_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        if self.reverse_sde.mh_sample is not None:
            if self.clipper_gen is not None:
                self.ema_model.score_clipper = self.clipper_gen
                reverse_sde = VEReverseSDE(
                    self.clipper_gen.wrap_grad_fxn(self.ema_model), 
                    self.noise_schedule, self.energy_function,
                    self.ema_model.MH_sample,
                    num_efficient_samples = self.num_efficient_samples
                )
                
            else:
                reverse_sde = VEReverseSDE(
                    self.ema_model, self.noise_schedule, self.energy_function,
                    self.ema_model.MH_sample,
                    num_efficient_samples = self.num_efficient_samples
                )
        else:
            if self.clipper_gen is not None:
                reverse_sde = VEReverseSDE(
                    self.clipper_gen.wrap_grad_fxn(self.ema_model), 
                    self.noise_schedule, self.energy_function,
                    num_efficient_samples = self.num_efficient_samples
                )
                
            else:
                reverse_sde = VEReverseSDE(
                    self.ema_model, self.noise_schedule, self.energy_function,
                    num_efficient_samples = self.num_efficient_samples
                )
        sample_start_time = time.time()
        # self.last_samples = self.generate_samples(
        #     reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
        # )
        self.last_samples = self.energy_function.sample_train_set(self.num_samples_to_generate_per_epoch, normalize=True)
        self.log(
            "val/sampling_time",
            time.time() - sample_start_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.last_energies = self.energy_function(self.last_samples)
        self.last_scores = self.energy_function.score(self.last_samples)
                
        # self.buffer.add(self.last_samples, self.last_energies.sum(-1))
        self.buffer.add(self.last_samples, self.last_energies, self.last_scores)  # add per-particle negative energy
        prefix = "val"
        
        self._log_energy_w2(prefix=prefix)
        self._log_data_w2(prefix=prefix)
        
        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix=prefix)
            self._log_dist_total_var(prefix=prefix)
        elif self.energy_function.dimensionality <= 2:
            self._log_data_total_var(prefix=prefix)

        # TODO: add MCMC refinement to buffer samples

    def _log_energy_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
            if self.clean_for_w2:
                generated_energies = self.energy_function(generated_samples).sum(-1)
                generated_samples = generated_samples[generated_energies > -100]
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, generated_energies = self.buffer.get_last_n_inserted(self.eval_batch_size)
        energies = self.energy_function(self.energy_function.normalize(data_set)).sum(-1)
        energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.sum(-1).cpu().numpy())#is there something wrong here? weird large number
        
        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
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

        def _grad_fxn(t, x):
            return self.clipped_grad_fxn(
                t,
                x,
                self.energy_function,
                self.noise_schedule,
                self.num_estimator_mc_samples,
            )

        reverse_sde = VEReverseSDE(_grad_fxn, self.noise_schedule)

        self.prior = self.partial_prior(device=self.device, scale=self.noise_schedule.h(1) ** 0.5)
        if not self.energy_function._can_normalize:
            self.cfm_prior = self.partial_prior(device=self.device, scale=self.cfm_prior_std)
        else:
            self.cfm_prior = self.partial_prior(device=self.device, scale=self.noise_schedule.h(1) ** 0.5)
        if self.init_from_prior:
            # init_states = self.prior.sample(self.num_init_samples)
            init_states = self.energy_function.sample_train_set(self.num_init_samples, normalize=True)
        else:
            init_states = self.generate_samples(
                None, self.num_init_samples, diffusion_scale=self.diffusion_scale
            )
        init_energies = self.energy_function(init_states)
        init_scores = self.energy_function.score(init_states)

        self.energy_function.log_on_epoch_end(
                init_states, init_energies,
                get_wandb_logger(self.loggers),
                epoch=self.iter_num
            )
        
        if isinstance(self.buffer, SimpleBufferVerbose) and self.buffer.buffer.energy is None:
            self.buffer.buffer = SimpleReplayDataVerbose(
                x=self.buffer.buffer.x,
                energy=torch.zeros((self.buffer.max_length, *init_energies.shape[1:])).to(init_energies.device),
                score=self.buffer.buffer.score
            )
        self.buffer.add(init_states, init_energies, init_scores)
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)