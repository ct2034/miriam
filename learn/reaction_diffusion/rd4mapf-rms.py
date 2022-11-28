#!/usr/bin/python3
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import laplace
from tqdm import tqdm

# src: https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb


def gray_scott_update(A, B, A_bg, B_bg, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """

    # Let's get the discrete Laplacians first
    LA = laplace(A, mode="constant", cval=A_bg)
    LB = laplace(B, mode="constant", cval=B_bg)

    # Now apply the update formula
    diff_A = (DA*LA - A*B**2 + f*(1-A)) * delta_t
    diff_B = (DB*LB + A*B**2 - (k+f)*B) * delta_t

    A += diff_A
    B += diff_B

    return A, B


def get_initial_configuration(N, random_influence=0.2):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """

    # We start with a configuration where on every grid cell
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N, N)) + \
        random_influence * np.random.random((N, N))

    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N, N))

    # Now let's add a disturbance in the center
    N2 = N//2
    radius = r = int(N/10.0)

    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25

    return A, B


def draw(d: Dict[str, Dict[str, np.ndarray]]):
    ncols = len(d)
    nrows = len(d[list(d.keys())[0]])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols*3, nrows*3))
    for i, (key, value) in enumerate(d.items()):
        for j, (key2, value2) in enumerate(value.items()):
            pos = axes[j, i].imshow(value2, cmap="hot")
            fig.colorbar(pos, ax=axes[j, i])
            axes[j, i].set_title(f"{key} {key2}")
            axes[j, i].axis("off")
    fig.tight_layout()


# update in time
delta_t = 1.0

# Diffusion coefficients
DA = 0.14
DB = 0.06

# define feed/kill rates
f = 0.035
k = 0.065

# grid size
N = 200

# simulation steps
N_simulation_steps = 10000

# "background" values for A and B
A_bg = 1.0
B_bg = 0.0

experiments = {
    "base": {
        "DA": 0.14,
        "DB": 0.06,
        "f": 0.035,
        "k": 0.065,
    },
    "more diffusion": {
        "DA": 0.16,
        "DB": 0.08,
        "f": 0.035,
        "k": 0.065,
    },
    "less diffusion": {
        "DA": 0.12,
        "DB": 0.04,
        "f": 0.035,
        "k": 0.065,
    },
}

results = {}

for experiment_name, experiment in experiments.items():
    A, B = get_initial_configuration(N)
    for i in tqdm(range(N_simulation_steps)):
        A, B = gray_scott_update(
            A, B, A_bg, B_bg, **experiment, delta_t=delta_t)
    results[experiment_name] = {"A": A, "B": B}

draw(results)
plt.savefig("learn/reaction_diffusion/rd4mapf-rms.png")
