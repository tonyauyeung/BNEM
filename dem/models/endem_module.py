import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning import LightningModule
from functools import partial

from .dem_module import *
from .components.energy_net_wrapper import EnergyNet
from .components.score_estimator import log_expectation_reward, estimate_grad_Rt
from .components.bootstrap_scheduler import BootstrapSchedule

class ENDEMLitModule(DEMLitModule):
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
        ais_steps: int = 5,
        ais_dt: float = 0.1,
        ais_warmup: int = 5e3,
        t0_regulizer_weight=0.5,
        bootstrap_schedule: BootstrapSchedule = None,
        bootstrap_warmup: int = 2e3,
        epsilon_train=1e-4,
    ) -> None:
            
            net = partial(EnergyNet, net=net)
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
            )
            self.t0_regulizer_weight = t0_regulizer_weight
            self.bootstrap_scheduler = bootstrap_schedule
            self.epsilon_train = epsilon_train
            self.bootstrap_warmup = bootstrap_warmup
            
    def forward(self, t: torch.Tensor, x: torch.Tensor, with_grad=False) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x, with_grad=with_grad)
    
    def energy_estimator(self, xt, t, num_samples, reduction=False):
        if self.ais_steps == 0 and self.iter_num < self.ais_warmup:
            return ais(xt, t, 
                       num_samples, self.ais_steps, 
                       self.noise_schedule, self.energy_function, 
                       dt=self.ais_dt)[0].detach()
        sigmas = self.noise_schedule.h(t).unsqueeze(1).sqrt()
        data_shape = list(xt.shape)[1:]
        noise = torch.randn(xt.shape[0], num_samples, *data_shape).to(xt.device)
        x0_t = noise * sigmas.unsqueeze(-1) + xt.unsqueeze(1)
        if reduction:
            energy_est = torch.logsumexp(self.energy_function(x0_t), dim=1) - torch.log(torch.tensor(num_samples)).to(xt.device)
            return energy_est
        return self.energy_function(x0_t)
    
    def sum_energy_estimator(self, e, num_samples):
        return torch.logsumexp(e, dim=1) - torch.log(torch.tensor(num_samples)).to(e.device) 
    
    
    def bootstrap_energy_estimator(
            self, 
            xt: torch.Tensor, 
            t: torch.Tensor, 
            u: torch.Tensor, 
            num_samples: list, 
            teacher_net: nn.Module,
            noise: Optional[torch.Tensor] = None
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
            teacher_out = teacher_net.forward_e(u, xu).reshape(-1, num_samples)
        #log_sum_exp = torch.logsumexp(teacher_out, dim=1) - torch.log(torch.tensor(num_samples, device=self.device))
        #return log_sum_exp
        return teacher_out
    
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
        
        return 1 / torch.nn.functional.l1_loss(energy_est, predicted_energy, reduction='none')
        
    
    def get_loss(self, times: torch.Tensor, 
                 samples: torch.Tensor, 
                 clean_samples: torch.Tensor,
                 train=False) -> torch.Tensor:
        
        self.iter_num += 1
        
        energy_est = self.energy_estimator(samples, times, self.num_estimator_mc_samples)
        if self.bootstrap_scheduler is not None and train and self.iter_num > self.bootstrap_warmup:
            i = self.bootstrap_scheduler.t_to_index(times.cpu())
            #i_tmp = i[torch.randint(i.shape[0], (1,))].item()
            #i = torch.full_like(i, i_tmp).long()
            t = self.bootstrap_scheduler.sample_t(i)
            u = self.bootstrap_scheduler.sample_t(i - 1)
            t = torch.clamp(t,min=self.epsilon_train).float().to(samples.device)
            u = torch.clamp(u,min=self.epsilon_train).float().to(samples.device)
            val_model = copy.deepcopy(self.net).to(samples.device).eval()


            u_confidence = self.bootstrap_confidence(clean_samples, u, 
                                                     self.num_estimator_mc_samples, 
                                                     val_model)
            t_confidence = self.bootstrap_confidence(clean_samples, 
                                                      t, 
                                                      self.num_estimator_mc_samples, 
                                                      val_model)

            sample_ratio = t_confidence / u_confidence
            sample_mc_prop_ratio = torch.rand(sample_ratio.shape, device=samples.device) 
            bootstrap_index = torch.where(sample_mc_prop_ratio < sample_ratio)[0]
            self.log(
                "bootstrap_accept_rate",
                bootstrap_index.shape[0] / sample_ratio.shape[0],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            bootstrap_energy_est = self.bootstrap_energy_estimator(samples[bootstrap_index], t[bootstrap_index], u[bootstrap_index], 
                                                                     self.num_estimator_mc_samples//5,
                                                                val_model)
            energy_est_full = self.sum_energy_estimator(energy_est, self.num_estimator_mc_samples)
            
            rand_index = torch.randint(0, self.num_estimator_mc_samples, (samples.shape[0], 4 * self.num_estimator_mc_samples//5))
            energy_est = torch.stack([energy_est[i, rand_index[i]] for i in range(energy_est.shape[0])], dim=0)
            boot_strap_energy_est = torch.cat([energy_est[bootstrap_index], bootstrap_energy_est], dim=1)
            energy_est_full[bootstrap_index] = self.sum_energy_estimator(boot_strap_energy_est,
                                                                         self.num_estimator_mc_samples)
            energy_est = energy_est_full
            
        else:
            energy_est = self.sum_energy_estimator(energy_est, self.num_estimator_mc_samples)

        
        predicted_energy = self.net.forward_e(times, samples)
        
        energy_clean = self.energy_function(clean_samples)

        predicted_energy_clean = self.net.forward_e(torch.zeros_like(times), clean_samples)
        
        
        error_norms = torch.nn.functional.l1_loss(energy_est, predicted_energy, reduction='none')
        error_norms_t0 = torch.nn.functional.l1_loss(energy_clean, predicted_energy_clean, reduction='none')
        
        #return error_norms + self.t0_regulizer_weight * error_norms_t0
        return self.lambda_weighter(times) ** 0.5 * error_norms  + \
             error_norms_t0 * self.t0_regulizer_weight
        
    
    def get_bootstrap_loss(self, times: torch.Tensor, 
                 samples: torch.Tensor, 
                 clean_samples: torch.Tensor) -> torch.Tensor:
        
        # we resample times for bootstrapping pairs
        t = torch.rand((samples.shape[0], ))
        i = self.bootstrap_scheduler.t_to_index(t)
        #i_tmp = i[torch.randint(i.shape[0], (1,))].item()
        #i = torch.full_like(i, i_tmp).long()
        t = self.bootstrap_scheduler.sample_t(i)
        u = self.bootstrap_scheduler.sample_t(i - 1)
        t = torch.clamp(t,min=self.epsilon_train).float().to(samples.device)
        u = torch.clamp(u,min=self.epsilon_train).float().to(samples.device)
        
        val_model = copy.deepcopy(self.net).to(samples.device).eval()
        
        energy_est = self.bootstrap_energy_estimator(samples, t, u, 
                                                     self.num_estimator_mc_samples//10, val_model)
        
        predicted_energy = self.net.forward_e(t, samples)
        
        
        energy_clean = self.energy_function(clean_samples)
        
        predicted_energy_clean = self.net.forward_e(torch.zeros_like(times), clean_samples)
        
        
        error_norms = torch.nn.functional.l1_loss(energy_est, predicted_energy, reduction='none')
        error_norms_t0 = torch.nn.functional.l1_loss(energy_clean, predicted_energy_clean, reduction='none')
        
        #return error_norms + self.t0_regulizer_weight * error_norms_t0
        return self.lambda_weighter(t) ** 0.5 * error_norms  + \
             error_norms_t0 * self.t0_regulizer_weight
        
        

        
