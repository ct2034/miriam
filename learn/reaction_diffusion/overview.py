#!/usr/bin/python3
from itertools import product
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# src: https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb


def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4*M
    L += np.roll(M, (0, -1), (0, 1))  # right neighbor
    L += np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L


def gray_scott_update(A, B, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """

    # Let's get the discrete Laplacians first
    LA = discrete_laplacian(A)
    LB = discrete_laplacian(B)

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

    # Now let's add four distrubances
    N4 = N//4
    N3 = N//4*3
    radius = r = int(N/10.0)

    for n1, n2 in product([N4, N3], repeat=2):
        A[n1-r:n1+r, n2-r:n2+r] = 0.50
        B[n1-r:n1+r, n2-r:n2+r] = 0.25

    return A, B


def draw(d: Dict[str, Dict[str, np.ndarray]]):
    ncols = len(d)
    nrows = len(d[list(d.keys())[0]])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols*3, nrows*3))
    for i, (key, value) in enumerate(d.items()):
        for j, (key2, value2) in enumerate(value.items()):
            axes[j, i].imshow(value2, cmap="gray")
            axes[j, i].set_title(f"{key} {key2}")
            axes[j, i].axis("off")
    fig.tight_layout()


if __name__ == "__main__":
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

    experiments = {
        "base": {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
        },
        "faster": {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.045,
            "k": 0.065,
        },
        "slower": {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.025,
            "k": 0.065,
        },
        "more kill": {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.075,
        },
        "less kill": {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.055,
        },
        "even less kill": {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.045,
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
            A, B = gray_scott_update(A, B, **experiment, delta_t=delta_t)
        results[experiment_name] = {"A": A, "B": B}

    draw(results)
    plt.savefig("learn/reaction_diffusion/overview.png")
    plt.show()
