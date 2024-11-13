import torch
from dem.energies.gmm_energy import *
import types

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import itertools
from PIL import Image, ImageChops


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

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    # Compute the difference between the input image and the solid background image
    diff = ImageChops.difference(im, bg)
    # Enhance the difference for better visibility
    diff = ImageChops.add(diff, diff, 2.0, -100)
    # Get the bounding box of the non-zero regions in the difference image
    bbox = diff.getbbox()
    # If a bounding box is found
    if bbox:
        # Keep the upper border by setting the top coordinate of bbox to 0
        bbox = (bbox[0], 0, bbox[2], bbox[3])
        # Crop the image according to the modified bounding box
        return im.crop(bbox)
    # If no bounding box is found, return the original image
    return im


def save_pil_image(img, file_name, dpi=300):
    # Calculate the size in inches
    img = trim(img)

    width_in = img.width / dpi
    height_in = img.height / dpi
    
    # Create a new image with the calculated size and white background
    new_img = Image.new("RGB", (int(width_in * dpi), int(height_in * dpi)), "white")
    
    # Paste the original image onto the new image
    new_img.paste(img, (0, 0))
    
    # Save the image
    new_img.save(file_name, dpi=(dpi, dpi), quality=95)


def plot_contours(log_prob_func,
                  ax: Optional[plt.Axes] = None,
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  grid_width_n_points: int = 20,
                  n_contour_levels: Optional[int] = None,
                  log_prob_min: float = -1000.0):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points).detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)


def plot_marginal_pair(samples: torch.Tensor,
                  ax: Optional[plt.Axes] = None,
                  marginal_dims: Tuple[int, int] = (0, 1),
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  alpha: float = 0.5,
                  color: str = 'orange'):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = torch.clamp(samples, bounds[0], bounds[1])
    samples = samples.cpu().detach()
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha, c=color)


def plot_gmm(gmm_energy, samples, name, plotting_bounds=(-1.4 * 40, 1.4 * 40), n=10000):
    if samples is None:
        name = 'gt'
        samples = samples = gmm.gmm.sample((n, ))
    else:
        samples = samples[:n]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

    gmm_energy.gmm.to("cpu")
    plot_contours(
        gmm_energy.gmm.log_prob,
        bounds=plotting_bounds,
        ax=ax,
        n_contour_levels=50,
        grid_width_n_points=200,
    )

    plot_marginal_pair(samples, ax=ax, bounds=plotting_bounds, alpha=0.1, color=color_maps[name])
    # ax.set_title(f"{name}")
    plt.axis('off')
    plt.title(name_maps[sampler], fontsize=60)
    plt.tight_layout()
    gmm_energy.gmm.to(gmm_energy.device)
    img = fig_to_image(fig)
    save_pil_image(img, f'plots/{name}_gmm.png', dpi=300)
    # img.save(f'plots/{name}_gmm.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    gmm = GMM()
    maps_sampler_path = { 
        'gt': '',
        'dem': '/home/ro352/rds/hpc-work/beidem/samples_file/gmm/gmm_dem.pt',
        'endem': '/home/ro352/rds/hpc-work/beidem/samples_file/gmm/gmm_endem.pt',
        'bendem': '/home/ro352/rds/hpc-work/beidem/samples_file/gmm/gmm_endem_bootstrap.pt',
        'dds': '/home/ro352/rds/hpc-work/beidem/samples_file/gmm/gmm_dds.pt',
        'fab': '/home/ro352/rds/hpc-work/beidem/samples_file/gmm/gmm_fab.pt',
        'pis': '/home/ro352/rds/hpc-work/beidem/samples_file/gmm/gmm_pis.pt',
    }
    for sampler in maps_sampler_path.keys():
        print(sampler)
        samples = torch.load(maps_sampler_path[sampler], map_location=torch.device('cpu')) * 50 if maps_sampler_path[sampler] else None
        plot_gmm(gmm, samples, sampler, n=10000)