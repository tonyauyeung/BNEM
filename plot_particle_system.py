import torch
from dem.energies.multi_double_well_energy import *
from dem.energies.lennardjones_energy import *
from dem.utils.data_utils import remove_mean
import types

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import itertools
import seaborn as sns


sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# color_maps = {
#     'gt': '#1f77b4',     # blue
#     'bendem': '#ff7f0e',    # orange
#     'endem': '#2ca02c',    # green
#     'dem': '#d62728',    # red
#     'fab': '#9467bd',    # purple
#     'pis': '#8c564b',  # brown
#     'dds': '#e377c2', # pink
# }


# name_maps = {
#     'gt': 'Ground Truth',
#     'bendem': 'BEnDEM (ours)',
#     'endem': 'EnDEM (ours)',
#     'dem': 'iDEM',
#     'fab': 'FAB', 
#     'pis': 'PIS',
#     'dds': 'DDS',
# }


def plot_interatomic_dist(energy_func, maps_sampler_samples, color_maps, name_maps, n=1000, alpha=0.5, saving_file=None):
    test_data_smaller = energy_func.sample_test_set(n)
    dist_test = energy_func.interatomic_dist(test_data_smaller).detach().cpu()
    if energy_func.n_particles == 55:
        bins = 50
        max_inter_dist = 7
        min_inter_dist = 0
    elif energy_func.n_particles == 13:
        bins = 100
        max_inter_dist = 6
        min_inter_dist = 0
    elif energy_func.n_particles == 4:
        bins = 100
        max_inter_dist = 9
        min_inter_dist = 0
    maps_sampler_distsamples = {}
    for sampler, samples in maps_sampler_samples.items():
        print(sampler)
        maps_sampler_distsamples[sampler] = energy_func.interatomic_dist(samples[:n]).detach().cpu() if samples is not None else None
    # maps_sampler_distsamples = {
    #     sampler: dw4_energy.interatomic_dist(samples[:10000]).detach().cpu() if samples is not None else None for sampler, samples in maps_sampler_samples.items()
    # }
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(
        dist_test.view(-1),
        bins=100,
        alpha=alpha,
        density=True,
        histtype="step",
        range=(min_inter_dist, max_inter_dist),
        linewidth=4,
        color=color_maps['gt'],
        label=name_maps['gt']
    )
    for sampler, dist_samples in maps_sampler_distsamples.items():
        if dist_samples is None:
            continue
        plt.hist(
            dist_samples.view(-1),
            bins=bins,
            alpha=alpha,
            density=True,
            histtype="step",
            linewidth=4,
            range=(min_inter_dist, max_inter_dist),
            color=color_maps[sampler],
            label=name_maps[sampler]
        )
    plt.legend(loc='upper right', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel(r'Interatomic Distance', fontsize=14)  # LaTeX in xlabel
    plt.ylabel(r'Probability Density', fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'plots/{energy_func.name}_interatomic_dist.png' if saving_file is None else saving_file, dpi=300, bbox_inches='tight')


def plot_energy(energy_func, maps_sampler_samples, color_maps, name_maps, n=1000, alpha=0.5, saving_file=None):
    plt.figure(figsize=(8, 6), dpi=300)
    bins = 100
    if energy_func.n_particles == 13:
        min_energy = -60
        max_energy = 0

    elif energy_func.n_particles == 55:
        min_energy = -380
        max_energy = -150

    elif energy_func.n_particles == 4:
        min_energy = -26
        max_energy = 0

    test_data_smaller = energy_func.sample_test_set(n)
    energy_test = -energy_func(test_data_smaller).detach().detach().cpu().sum(-1)
    
    maps_sampler_energies = {}
    for sampler, samples in maps_sampler_samples.items():
        print(sampler)
        maps_sampler_energies[sampler] = -energy_func(samples[:n]).detach().detach().cpu().sum(-1) if samples is not None else None
    plt.hist(
        energy_test.cpu(),
        bins=bins,
        density=True,
        alpha=alpha,
        range=(min_energy, max_energy),
        histtype="step",
        linewidth=4,
        color=color_maps['gt'],
        label=name_maps['gt'],
    )
    for sampler, energies in maps_sampler_energies.items():
        if energies is None:
            continue
        plt.hist(
            energies.cpu(),
            bins=bins,
            density=True,
            alpha=alpha,
            range=(min_energy, max_energy),
            histtype="step",
            linewidth=4,
            color=color_maps[sampler],
            label=name_maps[sampler],
        )
    plt.legend(loc='upper right', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel(r'$\mathcal{E}(x)$', fontsize=14)  # LaTeX in xlabel
    plt.ylabel(r'Probability Density', fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'plots/{energy_func.name}_energies.png' if saving_file is None else saving_file, dpi=300, bbox_inches='tight')

def get_dw4_data():
    dw4 = MultiDoubleWellEnergy(
        dimensionality = 8,
        n_particles = 4,
        data_path = "data/test_split_DW4.npy",
        data_path_train = "data/train_split_DW4.npy",
        data_path_val = "data/val_split_DW4.npy",
    )
    maps_sampler_path = { 
        'dem': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_dem.pt',
        'endem': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_endem.pt',
        'bendem': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_endem_bootstrap.pt',
        'dds': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_dds.pt',
        'fab': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_fab.pt',
        # 'pis': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_pis.pt',
    }
    return dw4, maps_sampler_path


def get_lj13_data():
    lj13 = LennardJonesEnergy(
        dimensionality=39,
        n_particles=13,
        data_path="data/test_split_LJ13-1000.npy",
        data_path_train="data/train_split_LJ13-1000.npy",
        data_path_val="data/test_split_LJ13-1000.npy"
    )
    maps_sampler_path = { 
        'dem': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_dem.pt',
        'endem': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_endem.pt',
        'bendem': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_endem_bootstrap.pt',
        # 'dds': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_dds.pt',
        # 'fab': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_fab.pt',
        # 'pis': '/home/ro352/rds/hpc-work/beidem/samples_file/dw4/dw4_pis.pt',
    }
    return lj13, maps_sampler_path


def get_lj55_data():
    lj55 = LennardJonesEnergy(
        dimensionality=165,
        n_particles=55,
        data_path="data/test_split_LJ55-1000-part1.npy",
        data_path_train="data/train_split_LJ55-1000-part1.npy",
        data_path_val="data/val_split_LJ55-1000-part1.npy"
    )
    maps_sampler_path = {
        # 'pis': ,
        # 'dds': ,
        # 'fab': ,
        'dem': '/home/ro352/rds/hpc-work/beidem/samples_file/lj55/lj55_dem.pt',
        'endem': '/home/ro352/rds/hpc-work/beidem/samples_file/lj55/lj55_endem.pt',
        'bendem': '/home/ro352/rds/hpc-work/beidem/samples_file/lj55/lj55_endem_bootstrap.pt',
    }
    return lj55, maps_sampler_path

def get_lj13_int_step_comp_data():
    lj13 = LennardJonesEnergy(
        dimensionality=39,
        n_particles=13,
        data_path="data/test_split_LJ13-1000.npy",
        data_path_train="data/train_split_LJ13-1000.npy",
        data_path_val="data/test_split_LJ13-1000.npy"
    )
    maps_sampler_path_1000 = { 
        'dem-1000': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_dem.pt',
        'endem-1000': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_endem.pt',
        'bendem-1000': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_endem_bootstrap.pt',
    }
    maps_sampler_path_100 = {
        'dem-100': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_dem_100.pt',
        'endem-100': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_endem_100.pt',
        'bendem-100': '/home/ro352/rds/hpc-work/beidem/samples_file/lj13/lj13_endem_bootstrap_100.pt',
    }

    return lj13, maps_sampler_path_1000, maps_sampler_path_100


def load_samples(energy_func, maps_sampler_path):
    maps_sampler_samples = {}
    for sampler in maps_sampler_path.keys():
        samples = torch.load(maps_sampler_path[sampler], map_location=torch.device('cpu')) if maps_sampler_path[sampler] else None
        maps_sampler_samples[sampler] = samples if samples is not None else None
    return maps_sampler_samples


if __name__ == '__main__':
    color_maps = {
        'gt': '#1f77b4',     # blue
        'bendem': '#ff7f0e',    # orange
        'endem': '#2ca02c',    # green
        'dem': '#d62728',    # red
        'fab': '#9467bd',    # purple
        'pis': '#8c564b',  # brown
        'dds': '#e377c2', # pink
    }

    name_maps = {
        'gt': 'Ground Truth',
        'bendem': 'BNEM (ours)',
        'endem': 'NEM (ours)',
        'dem': 'iDEM',
        'fab': 'FAB', 
        'pis': 'PIS',
        'dds': 'DDS',
    }
    alpha = 0.6

    # dw4, maps_sampler_path = get_dw4_data()
    # maps_sampler_samples = load_samples(dw4, maps_sampler_path)
    # n = 10000
    # plot_interatomic_dist(dw4, maps_sampler_samples, color_maps, name_maps, n)
    # plot_energy(dw4, maps_sampler_samples, color_maps, name_maps, n)

    # lj13, maps_sampler_path = get_lj13_data()
    # maps_sampler_samples = load_samples(lj13, maps_sampler_path)
    # n = 10000
    # plot_interatomic_dist(lj13, maps_sampler_samples, color_maps, name_maps, n, alpha)
    # plot_energy(lj13, maps_sampler_samples, color_maps, name_maps, n, alpha)

    lj55, maps_sampler_path = get_lj55_data()
    maps_sampler_samples = load_samples(lj55, maps_sampler_path)
    n = 3000
    plot_interatomic_dist(lj55, maps_sampler_samples, color_maps, name_maps, n, alpha)
    plot_energy(lj55, maps_sampler_samples, color_maps, name_maps, n, alpha)

    # lj13, maps_sampler_path_1000, maps_sampler_path_100 = get_lj13_int_step_comp_data()
    # maps_sampler_samples_1000 = load_samples(lj13, maps_sampler_path_1000)
    # maps_sampler_samples_100 = load_samples(lj13, maps_sampler_path_100)
    # color_maps_1000 = {}
    # color_maps_100 = {}
    # name_maps_1000 = {}
    # name_maps_100 = {}
    # for key in color_maps.keys():
    #     if key == 'gt':
    #         color_maps_1000[key] = color_maps[key]
    #         color_maps_100[key] = color_maps[key]
    #         name_maps_1000[key] = name_maps[key]
    #         name_maps_100[key] = name_maps[key]
    #     else:
    #         color_maps_1000[key+'-1000'] = color_maps[key]
    #         color_maps_100[key+'-100'] = color_maps[key]
    #         name_maps_1000[key+'-1000'] = name_maps[key]
    #         name_maps_100[key+'-100'] = name_maps[key]
    # n = 10000
    # plot_interatomic_dist(lj13, maps_sampler_samples_1000, color_maps_1000, name_maps_1000, n, alpha, 'plots/lj13_interdist_1000.png')
    # plot_interatomic_dist(lj13, maps_sampler_samples_100, color_maps_100, name_maps_100, n, alpha, 'plots/lj13_interdist_100.png')
    # plot_energy(lj13, maps_sampler_samples_1000, color_maps_1000, name_maps_1000, n, alpha, 'plots/lj13_energy_1000.png')
    # plot_energy(lj13, maps_sampler_samples_100, color_maps_100, name_maps_100, n, alpha, 'plots/lj13_energy_100.png')