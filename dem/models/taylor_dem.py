import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb

from .dem_module import *
from dem.models.components.prioritised_replay_buffer import SimpleBuffer, SimpleBufferVerbose, SimpleReplayDataVerbose
from dem.models.components.sampler.sampler import LangevinDynamics
from dem.models.components.sampler.dyn_mcmc_warp import DynSamplerWrapper


class TAYLORDEMLitModule(DEMLitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: SimpleBufferVerbose,
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
        ais_warmup: int = 0,
        ema_beta=0.95,
        ema_steps=0,
        iden_t=False,
        sample_noise=False,
        clean_for_w2=True,
        use_tweedie=False,
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
                ema_steps=ema_steps,
                iden_t=iden_t,
                sample_noise=sample_noise,
                clean_for_w2=clean_for_w2,
                use_tweedie=use_tweedie,
            )
            self.mcmc_step_size = 0.005
            self.mcmc_finetune_steps = 100
    
    def taylor_denoiser_estimator(self, t: torch.Tensor, xt: torch.Tensor):
        sigmas2 = self.noise_schedule.h(t).unsqueeze(-1)
        data_shape = xt.shape[1:]
        noise = torch.randn((xt.shape[0], self.num_estimator_mc_samples, *data_shape), device=xt.device)
        x0 = xt.clone().detach() + noise * sigmas2.sqrt()
        taylor_scores = self.energy_function.score(xt)
        weights = 1. + torch.matmul(taylor_scores.unsqueeze(1), xt.unsqueeze(1) - x0).squeeze(1)
        pdb.set_trace()
          


    def get_loss(self, times: torch.Tensor, samples: torch.Tensor, clean_samples: torch.Tensor, train=False) -> torch.Tensor:
        self.taylor_denoiser_estimator(times, samples)
        if train:
            self.iter_num += 1
        #clean samples is a placeholder for training on t=0 as regularizer
        if self.ais_steps == 0 or self.iter_num > self.ais_warmup:
            if not self.use_tweedie:
                estimated_score = estimate_grad_Rt(
                    times,
                    samples,
                    self.energy_function,
                    self.noise_schedule,
                    num_mc_samples=self.num_estimator_mc_samples,
                )
            else:
                estimated_score = estimate_score_tweedie(
                    times,
                    samples,
                    self.energy_function,
                    self.noise_schedule,
                    num_mc_samples=self.num_estimator_mc_samples,
                )
        else:
            estimated_score = ais(
                samples,
                times,
                self.num_estimator_mc_samples,
                self.ais_steps,
                self.noise_schedule,
                self.energy_function,
                dt=self.ais_dt,
            )

        if self.clipper is not None and self.clipper.should_clip_scores:
            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(
                    -1,
                    self.energy_function.n_particles,
                    self.energy_function.n_spatial_dim,
                )

            estimated_score = self.clipper.clip_scores(estimated_score)

            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(-1, self.energy_function.dimensionality)

        if self.score_scaler is not None:
            estimated_score = self.score_scaler.scale_target_score(estimated_score, times)

        predicted_score = self.forward(times, samples)

        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)

        return error_norms * self.lambda_weighter(times)