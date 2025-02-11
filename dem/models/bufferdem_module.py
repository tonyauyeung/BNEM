import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb

from .dem_module import *
from dem.models.components.prioritised_replay_buffer import SimpleBuffer, SimpleBufferVerbose, SimpleReplayDataVerbose
from dem.models.components.sampler.sampler import LangevinDynamics
from dem.models.components.sampler.dyn_mcmc_warp import DynSamplerWrapper


class BUFFERDEMLitModule(DEMLitModule):
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
    
    def buffer_score_estimator(self, xt, t):
        sigmas2 = self.noise_schedule.h(t).unsqueeze(-1).unsqueeze(-1)
        buffer_energy = self.buffer.buffer.energy.clone().detach()
        buffer_samples = self.buffer.buffer.x.clone().detach()
        buffer_scores = self.buffer.buffer.score.clone().detach()
        log_gaussian = -0.5 * ((xt[:, None, :] - buffer_samples[None, :, :]) ** 2).sum(dim=-1, keepdim=True) / sigmas2
        log_weights_unnormalized = buffer_energy + log_gaussian
        weights = torch.softmax(log_weights_unnormalized, dim=1)
        # return (buffer_scores * weights).sum(dim=1)
        mix_scores = (1. - sigmas2) * buffer_scores.unsqueeze(0) - (xt[:, None, :] - buffer_samples[None, :, :])
        return (mix_scores * weights).sum(dim=1)
    

    def get_loss(self, times: torch.Tensor, 
                 samples: torch.Tensor, 
                 clean_samples: torch.Tensor,
                 train=False) -> torch.Tensor:

        if train:
            self.iter_num += 1
        estimated_score = self.buffer_score_estimator(samples, times).detach()

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
    
    def on_train_epoch_start(self):
        lg = LangevinDynamics(x=self.buffer.buffer.x,
                                energy_func=lambda x: -self.energy_function(x).sum(-1),
                                step_size=self.mcmc_step_size,
                                mh=True,
                                device=self.buffer.device)
        self.mcmc_sampler = DynSamplerWrapper(lg, per_temp=False, target_acceptance_rate=0.6, alpha=0.25)
        for _ in range(self.mcmc_finetune_steps):
            new_samples, acc, _ = self.mcmc_sampler.sample()
        self.mcmc_acc_rate = acc
        new_samples = new_samples.clone().detach()
        new_energies = self.energy_function(new_samples)
        new_scores = self.energy_function(new_samples)
        self.buffer.add(new_samples, new_energies, new_scores)
        self.mcmc_step_size = self.mcmc_sampler.sampler.step_size
        self.train_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.EMA.step_ema(self.ema_model, self.net)
        self.log(
            "val/training_time",
            time.time() - self.train_start_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train/mcmc_acc_rate", self.mcmc_acc_rate, on_step=False, on_epoch=True, prog_bar=False,)
        if self.clipper is not None:
            reverse_sde = VEReverseSDE(
                self.clipper.wrap_grad_fxn(self.ema_model), 
                self.noise_schedule, self.energy_function
            )
        else:
            reverse_sde = VEReverseSDE(
                self.ema_model, self.noise_schedule, self.energy_function
            )
        sample_start_time = time.time()
        self.last_samples = self.generate_samples(
            reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
        )
        # self.last_samples = self.energy_function.sample_train_set(self.num_samples_to_generate_per_epoch, normalize=True)
        self.log(
            "val/sampling_time",
            time.time() - sample_start_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.last_energies = self.energy_function(self.last_samples)
        self.last_scores = self.energy_function.score(self.last_samples)
        self.buffer.add(self.last_samples, self.last_energies, self.last_scores)  # add per-particle negative energy
        prefix = "val"

        self._log_energy_w2(prefix=prefix)
        self._log_data_w2(prefix=prefix)
        
        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix=prefix)
            self._log_dist_total_var(prefix=prefix)
        elif self.energy_function.dimensionality <= 2:
            self._log_data_total_var(prefix=prefix)            
            

    def _log_energy_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
            if self.clean_for_w2:
                generated_energies = self.energy_function(generated_samples)
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
            init_states = self.prior.sample(self.num_init_samples)
            # init_states = self.energy_function.sample_train_set(self.num_init_samples, normalize=True)
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


    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        # convert to dict of tensors assumes [batch, ...]
        outputs = {
            k: torch.cat([dic[k] for dic in self.eval_step_outputs], dim=0)
            for k in self.eval_step_outputs[0]
        }

        unprioritized_buffer_samples, cfm_samples = self.buffer.buffer.x, None
        if self.nll_with_cfm:
            unprioritized_buffer_samples, _, _ = self.buffer.sample(
                self.eval_batch_size,
                prioritize=self.prioritize_cfm_training_samples,
            )

            cfm_samples = self.cfm_cnf.generate(
                self.cfm_prior.sample(self.eval_batch_size),
            )[-1]
            #with torch.no_grad():
            #    cfm_samples =  self.cfm_cnf.reverse_fn#(self.cfm_prior.sample(self.eval_batch_size))[-1]
            cfm_samples = self.energy_function.unnormalize(cfm_samples)
            self.energy_function.log_on_epoch_end(
                self.last_samples,
                self.last_energies,
                wandb_logger,
                unprioritized_buffer_samples=unprioritized_buffer_samples,
                cfm_samples=cfm_samples,
                replay_buffer=self.buffer,
                epoch=self.iter_num
            )
            
            #log training data
            train_samples = self.energy_function.sample_train_set(self.eval_batch_size)
            
            self.energy_function.log_samples(
                train_samples,
                wandb_logger,
                name="train",
            )

        else:
            if self.clipper_gen is not None:
                reverse_sde = VEReverseSDE(
                    self.clipper_gen.wrap_grad_fxn(self.ema_model),
                    # self.clipper_gen.wrap_grad_fxn(lambda t, x: self.buffer_score_estimator(x, t)),
                    self.noise_schedule, self.energy_function
                )
            else:
                reverse_sde = VEReverseSDE(
                    self.ema_model,
                    # lambda t, x: self.buffer_score_estimator(x, t),
                    self.noise_schedule, self.energy_function
                )
            generated_samples = self.generate_samples(
                reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
            )
            generated_energies = self.energy_function(generated_samples).sum(-1)
            # Only plot dem samples
            self.energy_function.log_on_epoch_end(
                # self.last_samples,
                # self.last_energies,
                generated_samples,
                generated_energies,
                wandb_logger,
                epoch=self.iter_num,
                unprioritized_buffer_samples=unprioritized_buffer_samples,
                replay_buffer=self.buffer
            )