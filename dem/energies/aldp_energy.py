import torch
import mdtraj
import os
import numpy as np
from PIL import Image

from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.energies.aldp_boltzmann_dist import get_aldp_target
from dem.energies.aldp_utils import evaluate_aldp



def load_data(target, data_path, num_data=1000000, device='cpu'):
    if data_path[-2:] == 'h5':
        mdtraj_traj = mdtraj.load(data_path)
        data = torch.tensor(mdtraj_traj.xyz).float().to(device).view(-1, 66)
        data = target.coordinate_transform.transform(data[:num_data])[0]
        data = target.coordinate_transform.inverse(data)[0]
    elif data_path[-2:] == 'pt':
        data = torch.load(data_path, map_location=device).float()
    return data


class AldpBoltzmannEnergy(BaseEnergyFunction):
    def __init__(
        self, 
        dimensionality: int,
        config: dict,
        test_data_path: str,
        val_data_path: str,
        device='cpu',
        plot_samples_epoch_period=5,
        data_normalization_factor=1.0
    ):
        self.target = get_aldp_target(config, device)
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        
        self.plot_samples_epoch_period = plot_samples_epoch_period
        self.data_normalization_factor = data_normalization_factor
        
        self.name = "aldp"
        self.device = device
        self.cur_epoch = 0
        
        super(AldpBoltzmannEnergy, self).__init__(dimensionality=dimensionality, is_molecule=False)

        # The following codes are used for ensuring the dihedral is from [-pi, pi]
        ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
        ncarts = self.target.coordinate_transform.transform.len_cart_inds
        dih_ind_ = self.target.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()
        ind = np.arange(60)
        ind = np.concatenate([ind[:3 * ncarts - 6], -np.ones(6, dtype=int), ind[3 * ncarts - 6:]])
        dih_ind = ind[dih_ind_]
        ind_circ = dih_ind[ind_circ_dih]
        self.dih_ind = dih_ind
        self.ind_circ = ind_circ

    def log_prob(self, x):
        # return self.target.log_prob(x)
        if x.ndim == 2:
            log_prob = self.target.log_prob(x) - (x[:, self.ind_circ].abs() > np.pi).any(-1).float()*1e8
        else:
            x_shape = x.shape
            x_reshaped = x.reshape(-1, 60)
            log_prob= self.target.log_prob(x_reshaped) - (x_reshaped[:, self.ind_circ].abs() > np.pi).any(-1).float()*1e8
            log_prob = log_prob.reshape(x_shape[:-1])
        return torch.where(log_prob.isnan(), 1e8, log_prob).float()
    
    def __call__(self, samples: torch.Tensor, smooth=None):
        return self.log_prob(samples).unsqueeze(-1)
    
    def energy(self, x, smooth=False):
        return self.log_prob(x).unsqueeze(-1)

    def score(self, x):
        if len(x.shape) > 2:
            batch_size = x.shape[:-1]
            x = x.view(-1, x.shape[-1])
        else:
            batch_size = None
        x = x.requires_grad_().to(self.device)
        energy = self(x).sum()
        energy.backward()
        # #clip the gradient
        # grad_ = torch.clip(x.grad, -5, 5)
        if batch_size is not None:
            grad_ = grad_.view(*batch_size, -1)
        return grad_

    def setup_test_set(self):
        test_set = load_data(self.target, self.test_data_path, device=self.device)
        return test_set
    
    def setup_val_set(self):
        val_set = load_data(self.target, self.test_data_path, device=self.device)
        return val_set
    
    def setup_train_set(self):
        train_set = load_data(self.target, self.test_data_path, device=self.device)
        return train_set
    
    def log_on_epoch_end(
            self, 
            # latest_samples, latest_energies, log_dir, epoch,
            latest_samples: torch.Tensor,
            latest_energies: torch.Tensor,
            wandb_logger: WandbLogger,
            unprioritized_buffer_samples=None,
            cfm_samples=None,
            replay_buffer=None,
            prefix: str = "",
            epoch=None
        ):
        log_dir = f'plot_aldp/epoch_{epoch}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        os.makedirs(log_dir + '/metrics', exist_ok=True)
        os.makedirs(log_dir + '/plots', exist_ok=True)
        #latest_samples = fill_unused_internal(latest_samples)
        #samples same number of samples from the test set
        if latest_samples is not None:
            sample_test_set = self.sample_test_set(len(latest_samples))
            #sample_test_set = fill_unused_internal(sample_test_set)
            evaluate_aldp(latest_samples, sample_test_set.to(latest_samples.device),
                        latest_energies, 
                        self.target.coordinate_transform,
                        iter=self.cur_epoch,
                        metric_dir=log_dir + '/metrics',
                        plot_dir=log_dir + '/plots')
            marginal_angle = Image.open(os.path.join(log_dir + '/plots', 'marginals_%s_%07i.png' % ("angle", self.cur_epoch + 1)))
            marginal_bond = Image.open(os.path.join(log_dir + '/plots', 'marginals_%s_%07i.png' % ("bond", self.cur_epoch + 1)))
            marginal_dih = Image.open(os.path.join(log_dir + '/plots', 'marginals_%s_%07i.png' % ("dih", self.cur_epoch + 1)))
            phi_psi = Image.open(os.path.join(log_dir + '/plots', '%s_%07i.png' % ("phi_psi", self.cur_epoch + 1)))
            ramachandran = Image.open(os.path.join(log_dir + '/plots', '%s_%07i.png' % ("ramachandran", self.cur_epoch + 1)))

            wandb_logger.log_image(f"marginal_angle", [marginal_angle])
            wandb_logger.log_image(f"marginal_bond", [marginal_bond])
            wandb_logger.log_image(f"marginal_dih", [marginal_dih])
            wandb_logger.log_image(f"phi_psi", [phi_psi])
            wandb_logger.log_image(f"ramachandran", [ramachandran])
            self.cur_epoch += 1