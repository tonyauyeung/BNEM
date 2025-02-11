import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

from .endem_module import *



class ANNEALNEMLitModule(ENDEMLitModule):
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
        t_unsqueeze = t.clone().detach().unsqueeze(-1).unsqueeze(-1)
        # Check for molecules
        if self.energy_function.is_molecule:
             gaussian_energy = 0.5 * x0_t.pow(2).reshape(x0_t.shape[0], x0_t.shape[1], self.energy_function.n_particles, -1).sum(dim=-1)
        else:
             gaussian_energy = 0.5 * x0_t.pow(2).sum(dim=-1, keepdim=True)
        energy_est = self.energy_function(x0_t, smooth=True) * (1 - t_unsqueeze) - gaussian_energy * t_unsqueeze
        if reduction:
            energy_est = torch.logsumexp(energy_est, dim=1) -\
                torch.log(torch.tensor(num_samples)).to(xt.device)
            return energy_est
        return energy_est