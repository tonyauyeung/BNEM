import copy

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
        ais_steps: int = 0,
        ais_dt: float = 0.1,
        ais_warmup: int = 5e3,
        t0_regulizer_weight=1,
        bootstrap_schedule: BootstrapSchedule = None,
        bootstrap_warmup: int = 2e3,
        bootstrap_mc_samples: int = 100,
        epsilon_train=1e-4,
        c_loss_weight=100,
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
                iden_t=False
            )
            self.t0_regulizer_weight = t0_regulizer_weight
            self.bootstrap_scheduler = bootstrap_schedule
            self.epsilon_train = epsilon_train
            self.bootstrap_warmup = bootstrap_warmup
            self.bootstrap_mc_samples = bootstrap_mc_samples
            assert self.num_estimator_mc_samples > self.bootstrap_mc_samples
            
            self.c_loss_weight = c_loss_weight
            
            if use_ema:
                self.net = EMAWrapper(self.net)
            
            
    def forward(self, t: torch.Tensor, x: torch.Tensor, with_grad=False) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x, with_grad=with_grad)
    
    def energy_estimator(self, xt, t, num_samples, reduction=False):
        if self.ais_steps != 0 and self.iter_num > self.ais_warmup:
            return ais(xt, t, 
                       num_samples, self.ais_steps, 
                       self.noise_schedule, self.energy_function, 
                       dt=self.ais_dt, mode='energy', reduction=False)
        sigmas = self.noise_schedule.h(t).unsqueeze(1).sqrt()
        data_shape = list(xt.shape)[1:]
        noise = torch.randn(xt.shape[0], num_samples, *data_shape).to(xt.device)
        x0_t = noise * sigmas.unsqueeze(-1) + xt.unsqueeze(1)
        energy_est = self.energy_function(x0_t)
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
            teacher_out = teacher_net.forward_e(u, xu).reshape(-1, num_samples)
        if reduction:    
            log_sum_exp = torch.logsumexp(teacher_out, dim=1) - torch.log(torch.tensor(num_samples, device=self.device))
            return log_sum_exp
        return teacher_out
    
    '''#old one that compute in batch difference
    def contrastive_loss(self, datapoints, predictions, targets):
        if datapoints.shape[0] == 0:
            return torch.tensor(0.).to(datapoints.device)
        #pred_dist = predictions[:, None] - predictions[None, :]
        #tar_dist = targets[:, None] - targets[None, :]
        #data_dist = torch.cdist(datapoints, datapoints)
        
        #pred_dist = remove_diag(pred_dist)
        #tar_dist = remove_diag(tar_dist)
        
        pred_dist =  predictions
        tar_dist = targets
              
        pred_dist = (pred_dist - pred_dist.mean()) / (pred_dist.std() + 1e-5)
        tar_dist = (tar_dist - tar_dist.mean()) / (tar_dist.std() + 1e-5)
        
        return - (tar_dist * pred_dist)
    '''
    
    def contrastive_loss(self, predictions, noised_predictions, targets):
        return torch.log(torch.abs(predictions - targets) / \
            (torch.clamp(predictions - noised_predictions, min=0.) + 1e-4))
        
    
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
        
        self.iter_num += 1
        
        
        energy_est = self.energy_estimator(samples, times, self.num_estimator_mc_samples)
        predicted_energy = self.net.forward_e(times, samples)
        
        dt = 1e-2 #TODO move to config
        
        noised_samples_dt = samples + (
                torch.randn_like(samples) * self.noise_schedule.g(times).unsqueeze(-1) * dt
            )

        if self.energy_function.is_molecule:
            noised_samples_dt = remove_mean(
                noised_samples_dt,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )
        
        predicted_energy_noised = self.net.forward_e(times, noised_samples_dt)
        '''
        if self.bootstrap_scheduler is not None and train and self.train_stage == 1:
            with torch.no_grad():
                t_loss = (self.sum_energy_estimator(energy_est, self.num_estimator_mc_samples) \
                          - predicted_energy).pow(2) * self.lambda_weighter(times)
                
                i = self.bootstrap_scheduler.t_to_index(times.cpu())
                u = self.bootstrap_scheduler.sample_t(i - 1)
                u = torch.clamp(u,min=self.epsilon_train).float().to(samples.device)
                
                
                
                u_energy_est = self.bootstrap_energy_estimator(samples, times, u,
                                                            self.num_estimator_mc_samples,
                                                            self.ema_model, reduction=True)
                u_predicted_energy = self.net.forward_e(u, samples) 
                u_loss = (u_energy_est - u_predicted_energy).pow(2) * self.lambda_weighter(u)

            
            bootstrap_index = torch.where(t_loss * (self.bootstrap_mc_samples -1) / self.bootstrap_mc_samples\
                                         > u_loss)[0]
            self.log(
                "bootstrap_accept_rate",
                bootstrap_index.shape[0] / t_loss.shape[0],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            bootstrap_energy_est = self.bootstrap_energy_estimator(samples[bootstrap_index], 
                                                                   times[bootstrap_index], u[bootstrap_index], 
                                                                     self.bootstrap_mc_samples,
                                                                self.ema_model)
            energy_est_full = self.sum_energy_estimator(energy_est, self.num_estimator_mc_samples)
            
            rand_index = torch.randint(0, self.num_estimator_mc_samples, 
                                       (samples.shape[0], 
                                        self.num_estimator_mc_samples - self.bootstrap_mc_samples))
            energy_est = torch.stack([energy_est[i, rand_index[i]] for i in range(energy_est.shape[0])], dim=0)
            bootstrap_energy_est = torch.cat([energy_est[bootstrap_index], bootstrap_energy_est], dim=1)
            energy_est_full[bootstrap_index] = self.sum_energy_estimator(bootstrap_energy_est,
                                                                         self.num_estimator_mc_samples)
            energy_est = energy_est_full
        
        else:
        '''
        energy_est = self.sum_energy_estimator(energy_est, self.num_estimator_mc_samples)
            
        
        energy_clean = self.energy_function(clean_samples)

        predicted_energy_clean = self.net.forward_e(torch.zeros_like(times), clean_samples)
        
        threshold = -1000.
        
        error_norms = torch.abs(torch.clamp(energy_est, min=threshold) \
            - torch.clamp(energy_est, min=threshold))

       
        error_norms_t0 = torch.abs(torch.clamp(energy_clean , min=threshold)\
            - torch.clamp(predicted_energy_clean, min=threshold))
        
        predicted_score = self.forward(times, samples, with_grad=True)
        predicted_score_noised = self.forward(times, noised_samples_dt, with_grad=True)
        estimated_score = estimate_grad_Rt(
                times,
                samples,
                self.energy_function,
                self.noise_schedule,
                num_mc_samples=self.num_estimator_mc_samples,
            )
        if self.energy_function.is_molecule:
            estimated_score = estimated_score.reshape(-1, self.energy_function.dimensionality)

        '''
        threshold = 1000
        clip_eff = torch.clamp(threshold / \
            torch.linalg.vector_norm(predicted_score), max=1.)
        predicted_score = clip_eff * predicted_score
        clip_eff = torch.clamp(threshold / \
            torch.linalg.vector_norm(estimated_score), max=1.)
        estimated_score = clip_eff * estimated_score
        
        error_norms_score = torch.abs(estimated_score - predicted_score) ** 2
        '''
        error_norms_score = self.contrastive_loss(predicted_score, predicted_score_noised, estimated_score)
        
        
        self.log(
                "energy_loss_t0",
                error_norms_t0.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        
        self.log(
                "energy_loss",
                error_norms.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        self.log(
                "score_loss",
                error_norms_score.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        '''
        c_loss = self.contrastive_loss(samples, 
                                       predicted_energy,  
                                       energy_est) * self.c_loss_weight
        c_loss_t0 = self.contrastive_loss(clean_samples, 
                                       predicted_energy_clean,  
                                       energy_clean) * self.c_loss_weight
        '''
        c_loss = self.contrastive_loss(predicted_energy, predicted_energy_noised,  energy_est)
        
        self.log(
                "contrast_loss",
                c_loss.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        '''
        self.log(
                "contrast_loss_t0",
                c_loss_t0.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        '''
        
        self.log(
            "largest energy",
            energy_est.min(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
        )
        
        self.log(
            "mean energy",
            energy_est.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
        )
        return (c_loss  + error_norms_score.sum(-1)) / (self.lambda_weighter(times) ** 0.5) + \
            error_norms_t0.mean() * self.t0_regulizer_weight
        
    
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
        
        
        energy_est = self.bootstrap_energy_estimator(samples, t, u, 
                                                     self.num_estimator_mc_samples//5, self.ema_model)
                                                     
        
        predicted_energy = self.net.forward_e(t, samples)
        
        
        energy_clean = self.energy_function(clean_samples)
        
        predicted_energy_clean = self.net.forward_e(torch.zeros_like(times), clean_samples)
        
        
        error_norms = torch.nn.functional.l1_loss(energy_est, predicted_energy, reduction='none')
        error_norms_t0 = torch.nn.functional.l1_loss(energy_clean, predicted_energy_clean, reduction='none')
        
        #return error_norms + self.t0_regulizer_weight * error_norms_t0
        return self.lambda_weighter(t) ** 0.5 * error_norms  + \
             error_norms_t0 * self.t0_regulizer_weight
        
        

        
