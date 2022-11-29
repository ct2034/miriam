#!/usr/bin/python3
import pickle as pkl
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import laplace
from tqdm import tqdm

# src: https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb


def gray_scott_update(A, B, A_bg, B_bg, mask, DA, DB, f, k, delta_t):
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
    A[mask] = A_bg
    B[mask] = B_bg

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


def bitmap_to_point_poses(bitmap: np.ndarray) -> np.ndarray:
    """
    Convert a bitmap to a list of point poses.
    """
    assert bitmap.ndim == 2

    point_poses = np.array([[1, 2]])
    return point_poses


def sim(gray_scott_update, get_initial_configuration, delta_t, N,
        N_simulation_steps, A_bg, B_bg, experiments):
    results: Dict[str, Dict[str, np.ndarray]] = {}

    mask = np.zeros((N, N), dtype=bool)
    mask[N//3:2*N//3, 0:N//6] = True

    for experiment_name, experiment in experiments.items():
        A, B = get_initial_configuration(N)
        for i in tqdm(range(N_simulation_steps)):
            A, B = gray_scott_update(
                A, B, A_bg, B_bg, mask, **experiment, delta_t=delta_t)
        results[experiment_name] = {"A": A, "B": B}

    pkl.dump(results, open("learn/reaction_diffusion/rd4mapf-rms.pkl", "wb"))


def draw():
    d = pkl.load(open("learn/reaction_diffusion/rd4mapf-rms.pkl", "rb"))
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
    plt.savefig("learn/reaction_diffusion/rd4mapf-rms.png")


def processing():
    d = pkl.load(open("learn/reaction_diffusion/rd4mapf-rms.pkl", "rb"))
    ncols = len(d)
    fig, axes = plt.subplots(nrows=2, ncols=ncols)
    for i, (experiment, data) in enumerate(d.items()):
        assert "A" in data.keys()
        bitmap = data["A"] < data["A"].mean()
        pos = axes[0, i].imshow(bitmap, cmap="gray")
        fig.colorbar(pos, ax=axes[0, i])
        axes[0, i].set_title(f"{experiment} bitmap")
        axes[0, i].axis("off")
        axes[1, i].set_title(f"{experiment} poses")
        axes[1, i].axis("off")
        point_poses = bitmap_to_point_poses(bitmap)
        axes[1, i].scatter(point_poses[:, 1], point_poses[:, 0])
    fig.tight_layout()
    plt.savefig("learn/reaction_diffusion/rd4mapf-rms-poses.png")


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

    # "background" values for A and B
    A_bg = 0.0
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

    # sim(gray_scott_update, get_initial_configuration,
    #     delta_t, N, N_simulation_steps, A_bg, B_bg, experiments)
    draw()
    processing()
