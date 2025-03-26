import copy
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning import LightningModule
from functools import partial
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from .dem_module import *
from .components.energy_net_wrapper import EnergyNet
from .components.score_estimator import log_expectation_reward, estimate_grad_Rt
from .components.bootstrap_scheduler import BootstrapSchedule
from .components.ema import EMA

def remove_diag(x):
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]
    mask = torch.eye(N, dtype=torch.bool)

    # Invert the mask to get the off-diagonal elements
    off_diag_mask = ~mask
    off_diag_elements = x[off_diag_mask]
    x = off_diag_elements.view(N, N - 1)
    return x



class NEMLitModule(DEMLitModule):
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
            
            net = partial(EnergyNet, net=net, 
                          noise_schedule=noise_schedule,
                          max_iter=mh_iter)
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
                iden_t=False,
                sample_noise=False
            )
            self.t0_regulizer_weight = t0_regulizer_weight
            self.bootstrap_scheduler = bootstrap_schedule
            self.epsilon_train = epsilon_train
            self.bootstrap_warmup = bootstrap_warmup
            self.bootstrap_mc_samples = bootstrap_mc_samples
            self.prioritize_warmup = prioritize_warmup
            assert self.num_estimator_mc_samples > self.bootstrap_mc_samples
            
            self.num_efficient_samples = num_efficient_samples
            self.reverse_sde = VEReverseSDE(self.net, 
                                                    self.noise_schedule, 
                                                    self.energy_function, None, num_efficient_samples)

            
            if use_ema:
                self.net = EMAWrapper(self.net)
            self.net.score_clipper = clipper_gen
            
            self.bootstrap_from_checkpoint = bootstrap_from_checkpoint
            
    def forward(self, t: torch.Tensor, x: torch.Tensor, with_grad=False) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x, with_grad=with_grad)
    
    def energy_estimator(self, xt, t, num_samples, reduction=False):
        if self.ais_steps != 0 and self.iter_num < self.ais_warmup:
            return ais(xt, t, 
                       num_samples, self.ais_steps, 
                       self.noise_schedule, self.energy_function, 
                       dt=self.ais_dt, mode='energy', reduction=reduction)
        sigmas = self.noise_schedule.h(t).unsqueeze(1).sqrt()
        data_shape = list(xt.shape)[1:]
        noise = torch.randn(xt.shape[0], num_samples, *data_shape).to(xt.device)
        x0_t = noise * sigmas.unsqueeze(-1) + xt.unsqueeze(1)
        energy_est = self.energy_function(x0_t, smooth=True)
        if reduction:
            energy_est = torch.logsumexp(energy_est, dim=1) -\
                torch.log(torch.tensor(num_samples)).to(xt.device)
            return energy_est
        return energy_est
    
    def sum_energy_estimator(self, e, num_samples):
        return torch.logsumexp(e, dim=1) - torch.log(torch.tensor(num_samples)).to(e.device) 
    
    
    def bootstrap_energy_estimator(
            self, 
            xt: torch.Tensor, 
            t: torch.Tensor, 
            u: torch.Tensor, 
            num_samples: list, 
            teacher_net: nn.Module,
            noise: Optional[torch.Tensor] = None,
            reduction=False
        ) -> torch.Tensor:
        """
        Bootstrappingly estimate the energy at time t based on the energy at time u.
        """
        #assert torch.all(t >= (u-self.epsilon_train))
        if torch.all(u <= self.epsilon_train):
            """use the original estimator when u=0 --> t"""
            return self.energy_estimator(xt, t, num_samples)
        
        sigma_t = self.noise_schedule.h(t).unsqueeze(1).to(xt.device).sqrt()
        sigma_u = self.noise_schedule.h(u).unsqueeze(1).to(xt.device).sqrt()
        data_shape = list(xt.shape)[1:]
        if noise is None:
            noise = torch.randn(xt.shape[0], num_samples, *data_shape, device=self.device)
    
        xu = noise * torch.sqrt(sigma_t ** 2 - sigma_u ** 2).unsqueeze(-1) + xt.unsqueeze(1)
        u = u.tile(num_samples)
        xu = xu.flatten(0, 1)
        with torch.no_grad():
            teacher_out = teacher_net.forward_e(u, xu).reshape(-1, num_samples, self.energy_function.n_particles)
        if reduction:    
            log_sum_exp = torch.logsumexp(teacher_out, dim=1) - torch.log(torch.tensor(num_samples, device=self.device))
            return log_sum_exp
        return teacher_out
    

    def contrastive_loss(self, predictions, targets):
              
        pred_dist = (predictions - predictions.mean()) / (predictions.std() + 1e-5)
        tar_dist = (targets - targets.mean()) / (targets.std() + 1e-5)
        
        return - (tar_dist * pred_dist)
    
    
    def norm_loss(self, predictions, noised_predictions, targets):
        if len(predictions.shape) == 1:
            return torch.log((predictions - targets).pow(2) / \
            ((predictions.unsqueeze(1) - noised_predictions).pow(2).mean(1).detach() + 1e-4))
        elif len(predictions.shape) == 2:
            return torch.log(torch.linalg.vector_norm(predictions - targets, dim=-1) / \
            (torch.linalg.vector_norm(predictions.unsqueeze(1) - noised_predictions, dim=-1).mean(1).detach() + 1e-4))
        
    
    @torch.no_grad()
    def bootstrap_confidence(self, 
                             x0: torch.Tensor, 
                             t: torch.Tensor,
                             num_samples: int,
                             teacher_net: nn.Module,
                             noise: Optional[torch.Tensor] = None):
        
        sigma_t = self.noise_schedule.h(t).to(x0.device).sqrt()
        if noise is None:
            noise = torch.randn(x0.shape, device=self.device)
        xt = x0 + noise * sigma_t.unsqueeze(-1)
        predicted_energy = teacher_net.forward_e(t, xt)
        energy_est = self.energy_estimator(xt, t, num_samples, reduction=True)
        
        return (energy_est - predicted_energy).pow(2)
    
    
    def get_loss(self, times: torch.Tensor, 
                 samples: torch.Tensor, 
                 clean_samples: torch.Tensor,
                 train=False) -> torch.Tensor:
        
        if self.iter_num > 0:
            self.reverse_sde = VEReverseSDE(self.net, self.noise_schedule, 
                                            self.energy_function,self.net.MH_sample)
        energy_est = self.energy_estimator(samples, times, self.num_estimator_mc_samples).detach()
        predicted_energy = self.net.forward_e(times, samples)
        
        bootstrap_stage = 1 if self.bootstrap_from_checkpoint else 0
        
        should_bootstrap = (self.bootstrap_scheduler is not None and train and self.train_stage == bootstrap_stage)
        if should_bootstrap:
            self.iden_t = True
            with torch.no_grad():
                t_loss = (self.sum_energy_estimator(energy_est, self.num_estimator_mc_samples) \
                          - predicted_energy).pow(2).sum(-1) * self.lambda_weighter(times)
                
                i = self.bootstrap_scheduler.t_to_index(times.cpu())
                u = self.bootstrap_scheduler.sample_t(i - 1)
                u = torch.clamp(u,min=self.epsilon_train).float().to(samples.device)
                times = torch.clamp(times,min=self.epsilon_train).float().to(samples.device)
                
                u_samples = clean_samples + torch.randn_like(clean_samples) * self.noise_schedule.h(u).sqrt().unsqueeze(-1)
                u_energy_est = self.energy_estimator(u_samples, u, 
                                                     self.num_estimator_mc_samples, 
                                                     reduction=True)
                u_predicted_energy = self.net.forward_e(u, u_samples)
                u_loss = (u_energy_est - u_predicted_energy).pow(2).sum(-1) / self.lambda_weighter(u)
            
            bootstrap_index = torch.where(t_loss * (self.bootstrap_mc_samples -1) / self.bootstrap_mc_samples\
                                         > u_loss)[0]
            self.log(
                "bootstrap_accept_rate",
                bootstrap_index.shape[0] / t_loss.shape[0],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "u_loss",
                u_loss.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            
            bootstrap_energy_est = self.bootstrap_energy_estimator(samples[bootstrap_index], 
                                                                   times[bootstrap_index], u[bootstrap_index], 
                                                                     self.bootstrap_mc_samples,
                                                                self.ema_model)
            energy_est_full = self.sum_energy_estimator(energy_est, 
                                                        self.num_estimator_mc_samples)
            
            rand_index = torch.randint(0, self.num_estimator_mc_samples, 
                                       (samples.shape[0], 
                                        self.num_estimator_mc_samples - self.bootstrap_mc_samples))
            energy_est = torch.stack([energy_est[i, rand_index[i]] for i in range(energy_est.shape[0])], dim=0)
            bootstrap_energy_est = torch.cat([energy_est[bootstrap_index],
                                              bootstrap_energy_est], 
                                             dim=1)
            energy_est_full[bootstrap_index] = self.sum_energy_estimator(bootstrap_energy_est,
                                                                         self.num_estimator_mc_samples)
            energy_est = energy_est_full
        
        else:
        
            energy_est = self.sum_energy_estimator(energy_est, 
                                                   self.num_estimator_mc_samples)
            
        
        energy_clean = self.energy_function(clean_samples, smooth=True)

        predicted_energy_clean = self.net.forward_e(torch.zeros_like(times), 
                                                    clean_samples)
        energy_error_norm = torch.abs(predicted_energy - energy_est).pow(2)
        error_norms_t0 = torch.abs(energy_clean - predicted_energy_clean).pow(2)

        
        
        self.log(
                "energy_loss_t0",
                error_norms_t0.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
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
        
        #if self.iter_num == self.prioritize_warmup:
        #    self.buffer.prioritize = False
        
        self.log(
            "largest energy",
            energy_clean.min(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
        )
        
        self.log(
            "mean energy",
            energy_clean.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
        )
        
        full_loss = (self.t0_regulizer_weight * error_norms_t0 + energy_error_norm * self.lambda_weighter(times)).sum(-1)
        if should_bootstrap:
            full_loss += u_loss
        
        return full_loss
    
    def integrate(
        self,
        reverse_sde: VEReverseSDE = None,
        samples: torch.Tensor = None,
        reverse_time=True,
        return_full_trajectory=False,
        diffusion_scale=1.0,
        no_grad=True,
        negative_time=False,
    ) -> torch.Tensor:
        trajectory = integrate_sde(
            reverse_sde or self.reverse_sde,
            samples,
            self.num_integration_steps,
            self.energy_function,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
            negative_time=negative_time,
            num_negative_time_steps=self.hparams.num_negative_time_steps,
            metroplolis_hasting=(self.reverse_sde.mh_sample is not None)
        )
        if return_full_trajectory:
            return trajectory
        
        return trajectory[-1]
    
    def on_train_epoch_start(self):
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
        self.last_samples = self.generate_samples(
            reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
        )
        self.log(
            "val/sampling_time",
            time.time() - sample_start_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.last_energies = self.energy_function(self.last_samples)
                
        self.buffer.add(self.last_samples, self.last_energies.sum(-1))
        prefix = "val"

        
        self._log_energy_w2(prefix=prefix)
        self._log_data_w2(prefix=prefix)
        
        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix=prefix)
            self._log_dist_total_var(prefix=prefix)
        elif self.energy_function.dimensionality <= 2:
            self._log_data_total_var(prefix=prefix)
        if self.bootstrap_scheduler is not None:
            sample_path = f"traj_data/bnem_{self.energy_function.name}"
        else:
            sample_path = f"traj_data/nem_{self.energy_function.name}"
        os.makedirs(sample_path, exist_ok=True)
        torch.save(self.last_samples.detach().cpu().float(), os.path.join(sample_path, f"samples_{self.iter_num}.pt"))