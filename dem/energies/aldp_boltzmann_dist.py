import torch
from torch import nn
import numpy as np

import boltzgen as bg
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
import mdtraj
import tempfile

from typing import Optional, Dict

import abc
import yaml

class TargetDistribution(abc.ABC):

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn = None,
                            batch_size: Optional[int] = None) -> Dict:
        """
        Check performance metrics using samples & log weights from the model, as well as it's
        probability density function (if defined).
        Args:
            samples: Samples from the trained model.
            log_w: Log importance weights from the trained model.
            log_q_fn: Log probability density function of the trained model, if defined.
            batch_size: If performance metrics are aggregated over many points that require network
                forward passes, batch_size ensures that the forward passes don't overload GPU
                memory by doing all the points together.

        Returns:
            info: A dictionary of performance measures, specific to the defined
            target_distribution, that evaluate how well the trained model approximates the target.
        """
        raise NotImplementedError


    def sample(self, shape):
        raise NotImplementedError



class AldpBoltzmann(nn.Module, TargetDistribution):
    def __init__(self, data_path=None, temperature=1000, energy_cut=1.e+8,
                 energy_max=1.e+20, n_threads=4, transform='internal',
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2},
                 lj_smooth=False,
                 env='vacuum'):
        """
        Boltzmann distribution of Alanine dipeptide
        :param data_path: Path to the trajectory file used to initialize the
            transformation, if None, a trajectory is generated
        :type data_path: String
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        :param transform: Which transform to use, can be mixed or internal
        :type transform: String
        """
        super(AldpBoltzmann, self).__init__()

        # Define molecule parameters
        ndim = 66
        if transform == 'mixed':
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19])
            ]
            cart_indices = [6, 8, 9, 10, 14]
        elif transform == 'internal':
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (9, [8, 6, 4]),
                (10, [8, 6, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19])
            ]
            cart_indices = [8, 6, 14]

        # System setup
        if env == 'vacuum':
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError('This environment is not implemented.')

        # Smooth the Lennard-Jones potential if necessary
        self.lj_smooth = lj_smooth
        if lj_smooth:
            system = LJ_smooth(system)

        sim = app.Simulation(system.topology, system.system,
                             mm.LangevinIntegrator(temperature * unit.kelvin,
                                                   1. / unit.picosecond,
                                                   1. * unit.femtosecond),
                             mm.Platform.getPlatformByName('Reference'))

        # Generate trajectory for coordinate transform if no data path is specified
        if data_path is None:
            sim = app.Simulation(system.topology, system.system,
                                mm.LangevinIntegrator(temperature * unit.kelvin,
                                                      1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                                platform=mm.Platform.getPlatformByName('Reference'))
            sim.context.setPositions(system.positions)
            sim.minimizeEnergy()
            state = sim.context.getState(getPositions=True)
            position = state.getPositions(True).value_in_unit(unit.nanometer)
            tmp_dir = tempfile.gettempdir()
            data_path = tmp_dir + '/aldp.pt'
            torch.save(torch.tensor(position.reshape(1, 66).astype(np.float64)), data_path)

            del (sim)

        if data_path[-2:] == 'h5':
            # Load data for transform
            traj = mdtraj.load(data_path)
            traj.center_coordinates()

            # superpose on the backbone
            ind = traj.top.select("backbone")
            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

            # Gather the training data into a pytorch Tensor with the right shape
            transform_data = traj.xyz
            n_atoms = transform_data.shape[1]
            n_dim = n_atoms * 3
            transform_data_npy = transform_data.reshape(-1, n_dim)
            transform_data = torch.from_numpy(transform_data_npy.astype("float64"))
        elif data_path[-2:] == 'pt':
            transform_data = torch.load(data_path)
        else:
            raise NotImplementedError('Loading data or this format is not implemented.')

        # Set distribution
        self.coordinate_transform = bg.flows.CoordinateTransform(transform_data,
                                        ndim, z_matrix, cart_indices, mode=transform,
                                        ind_circ_dih=ind_circ_dih, shift_dih=shift_dih,
                                        shift_dih_params=shift_dih_params,
                                        default_std=default_std)

        if n_threads > 1:
            self.p = bg.distributions.TransformedBoltzmannParallel(system,
                            temperature, energy_cut=energy_cut, energy_max=energy_max,
                            transform=self.coordinate_transform, n_threads=n_threads)
        else:
            self.p = bg.distributions.TransformedBoltzmann(sim.context,
                            temperature, energy_cut=energy_cut, energy_max=energy_max,
                            transform=self.coordinate_transform)

    def log_prob(self, x: torch.tensor):
        return self.p.log_prob(x)

    def performance_metrics(self, samples, log_w, log_q_fn, batch_size):
        return {}



# def get_aldp_target(config_path, device):

#     with open(config_path, 'r') as stream:
#         config = yaml.load(stream, yaml.FullLoader)

#     # Target distribution
#     transform_mode = 'mixed' if not 'transform' in config['system'] \
#         else config['system']['transform']
#     shift_dih = False if not 'shift_dih' in config['system'] \
#         else config['system']['shift_dih']
#     env = 'vacuum' if not 'env' in config['system'] \
#         else config['system']['env']
#     ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
#     target = AldpBoltzmann(data_path=config['data']['transform'],
#                            temperature=config['system']['temperature'],
#                            energy_cut=config['system']['energy_cut'],
#                            energy_max=config['system']['energy_max'],
#                            n_threads=config['system']['n_threads'],
#                            transform=transform_mode,
#                            ind_circ_dih=ind_circ_dih,
#                            shift_dih=shift_dih,
#                            lj_smooth=config['system']['lj_smooth'],
#                            env=env)
#     target = target.to(device)

#     return target
def get_aldp_target(config, device):
    # Target distribution
    transform_mode = 'mixed' if not 'transform' in config['system'] \
        else config['system']['transform']
    shift_dih = False if not 'shift_dih' in config['system'] \
        else config['system']['shift_dih']
    env = 'vacuum' if not 'env' in config['system'] \
        else config['system']['env']
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    target = AldpBoltzmann(data_path=config['data']['transform'],
                           temperature=config['system']['temperature'],
                           energy_cut=config['system']['energy_cut'],
                           energy_max=config['system']['energy_max'],
                           n_threads=config['system']['n_threads'],
                           transform=transform_mode,
                           ind_circ_dih=ind_circ_dih,
                           shift_dih=shift_dih,
                           lj_smooth=config['system']['lj_smooth'],
                           env=env)
    target = target.to(device)

    return target


def LJ_smooth(system):
    for i in range(system.system.getNumForces()):
        force = system.system.getForce(i)

        if isinstance(force, mm.NonbondedForce):
            
            param_lambda = 0.8
            param_alpha = 0.8
            param_m = 2
            param_n = 1.0
            
            custom_lj_force = mm.CustomNonbondedForce(
                "4*epsilon*lambda^n*((alpha*(1 - lambda)^m + (r/sigma)^6)^(-2) - (alpha*(1 - lambda)^m + (r/sigma)^6)^(-1)); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); lambda=0.5*(lambda1+lambda2); m=0.5*(m1+m2); n=0.5*(n1+n2); alpha=0.5*(alpha1+alpha2)"
            )
            custom_lj_force.addPerParticleParameter("sigma")
            custom_lj_force.addPerParticleParameter("epsilon")
            custom_lj_force.addPerParticleParameter("lambda")
            custom_lj_force.addPerParticleParameter("alpha")
            custom_lj_force.addPerParticleParameter("m")
            custom_lj_force.addPerParticleParameter("n")
            
            # Disable LJ terms in the original NonbondedForce
            for i in range(force.getNumParticles()):
                charge, sigma, epsilon = force.getParticleParameters(i)
                
                # Retain only the charge term
                force.setParticleParameters(i, charge, 0.0 * sigma, 0.0 * epsilon)  # Zero out LJ terms
                
                # Add LJ parameters to the CustomNonbondedForce
                custom_lj_force.addParticle([sigma, epsilon, param_lambda, param_alpha, param_m, param_n])

            # Copy exceptions to CustomNonbondedForce
            for i in range(force.getNumExceptions()):
                p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                custom_lj_force.addExclusion(p1, p2)  # Exclude exceptions from LJ smoothing

            # Set cutoff and boundary conditions for CustomNonbondedForce
            cutoff_distance = force.getCutoffDistance()
            custom_lj_force.setCutoffDistance(cutoff_distance)
            custom_lj_force.setNonbondedMethod(force.getNonbondedMethod())
            custom_lj_force.setUseSwitchingFunction(True)
            custom_lj_force.setSwitchingDistance(0.75*cutoff_distance)

            # Add the CustomNonbondedForce to the system
            system.system.addForce(custom_lj_force)
            
            break
        
    return system