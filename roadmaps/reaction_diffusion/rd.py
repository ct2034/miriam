#!/usr/bin/python3
import pickle as pkl
from itertools import product
from random import Random
from typing import Tuple

import cv2
import numpy as np
import torch
from bresenham import bresenham
from matplotlib import pyplot as plt
from scipy.ndimage import laplace
from sklearn.cluster import AgglomerativeClustering  # scikit-learn
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm

from definitions import MAP_IMG

# src:
# https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb


def gray_scott_update(A, B, A_bg, B_bg, mask, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients `DA` and `DB`, as well as feed rate `f` and
    kill rate `k`.
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


def get_initial_configuration(N, random_influence=0.2, rng: Random = Random()
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize a concentration configuration. `N` is the side length
    of the (`N` x `N`)-sized grid.
    `random_influence` describes how much noise is added.
    """
    np.random.seed(rng.randint(0, 2**32))

    # We start with a configuration where on every grid cell
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N, N)) + \
        random_influence * np.random.random((N, N))

    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N, N))

    # Now let's add four distrubances
    N4 = N//4
    N3 = N//4*3
    r = int(N/10.0)

    for n1, n2 in product([N4, N3], repeat=2):
        A[n1-r:n1+r, n2-r:n2+r] = 0.50
        B[n1-r:n1+r, n2-r:n2+r] = 0.25

    return A, B


def bitmap_to_point_poses(bitmap: np.ndarray) -> np.ndarray:
    """
    Convert a bitmap to a list of point poses.
    """
    assert bitmap.ndim == 2
    assert bitmap.shape[0] == bitmap.shape[1]
    width = bitmap.shape[0]

    int_poses = []
    for x, y in product(range(bitmap.shape[0]), repeat=2):
        if bitmap[x, y]:
            int_poses.append((x, y))

    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.1, linkage="single")
    model.fit(int_poses)
    # return model.children_
    y_predict = model.fit_predict(int_poses)
    # ...
    clf = NearestCentroid()
    clf.fit(int_poses, y_predict)
    points = clf.centroids_ / width
    return np.unique(points, axis=0)


def is_free(bitmap, ax, ay, bx, by):
    """Check if a line between two points is free in a `bitmap`."""
    for x, y in bresenham(ax, ay, bx, by):
        if not bitmap[x, y]:
            return False
    return True


def find_radius(point_poses, bitmap):
    """Find the radius of the circles centred around `point_poses` in the
    `bitmap`."""
    radiuss = [0] * len(point_poses)
    for i_p, point_pose in enumerate(point_poses):
        c_x, c_y = (point_pose * bitmap.shape[0]).astype(int)
        for x, y in product(
            range(c_x-10, c_x+10),
            range(c_y-10, c_y+10)
        ):
            if x < 0 or y < 0 or x >= bitmap.shape[0] or y >= bitmap.shape[0]:
                continue
            if is_free(bitmap, x, y, c_x, c_y):
                radiuss[i_p] = max(radiuss[i_p], np.linalg.norm(
                    np.array([x, y]) - point_pose*bitmap.shape[0]))
    return np.mean(radiuss)


def poses_to_reaction_diffusion(
        point_poses: np.ndarray,
        A_min_max: Tuple[float, float], B_min_max: Tuple[float, float],
        width: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Regenerate A and B for reaction diffusion by making circles around
    the `point_poses`."""
    A_bg = A_min_max[1]
    A_fg = A_min_max[0] + .3
    B_bg = B_min_max[0]
    B_fg = B_min_max[1] - .1
    A = (np.random.rand(width, width) / 4 + 3 / 4) * A_bg
    B = (np.random.rand(width, width) / 4 + 3 / 4) * B_bg
    for i_p, point_pose in enumerate(point_poses):
        c_x, c_y = (point_pose * width).astype(int)
        for x, y in product(
            range(c_x-10, c_x+10),
            range(c_y-10, c_y+10)
        ):
            if np.linalg.norm(np.array([x, y]) - point_pose*width) < radius:
                A[x, y] = A_fg
                B[x, y] = B_fg
    return A, B


def draw(file: str):
    """Draw the reaction diffusion A and B from the `file`."""
    d = pkl.load(open(f"{file}.pkl", "rb"))
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
    plt.savefig(f"{file}.png")


def reaction_difussion_to_bitmap(B):
    """Convert a reaction diffusion data B to a binary bitmap."""
    bitmap = B > B.mean() * 1.7
    return bitmap


def plot_bitmap_and_poses(ax, bitmap, point_poses, title=""):
    """Plot a `bitmap` and the `point_poses` on the `ax`."""
    ax.imshow(bitmap, cmap="gray")
    ax.set_title(f"{title} bitmap, poses")
    ax.axis("off")
    ax.scatter(
        point_poses[:, 1] * bitmap.shape[0],
        point_poses[:, 0] * bitmap.shape[0],
        s=20,
        c="red",
        marker=".",
        alpha=0.9)


def sample_points_reaction_diffusion(
        n: int,
        map_img: MAP_IMG,
        rng: Random) -> Tuple[torch.Tensor, np.ndarray]:
    """Sample roughly `n` random points (0 <= x <= 1) from a map using
    reaction-diffusion."""
    alpha = 0.5
    n_searches = 10
    for i in range(n_searches):
        point_poses: np.ndarray = np.empty(shape=(0, 2))
        try:
            delta_t, experiment, N_simulation_steps, size = get_experiments(
                alpha)
            mask = cv2.resize(np.array(map_img).astype(
                np.float32), (size, size))
            mask = mask < 255
            assert mask.shape[0] == mask.shape[1], "Mask must be square."
            A, B = get_initial_configuration(mask.shape[0], rng=rng)
            A_bg = 0.0
            B_bg = 0.0
            for i in tqdm(range(N_simulation_steps)):
                A, B = gray_scott_update(
                    A, B, A_bg, B_bg, mask, **experiment, delta_t=delta_t)
            bitmap = reaction_difussion_to_bitmap(B)
            point_poses = bitmap_to_point_poses(bitmap)
            print(f"Found {len(point_poses)} points, when {n} were requested.")
        except Exception as e:
            print(f"Exception: {e}")
            assert len(point_poses) != 0, "No points found."
        if abs(len(point_poses) - n) < 0.2 * n:
            break
        elif len(point_poses) > n:
            alpha *= (1 + 0.0003 * (n_searches - i) / n_searches)
        else:
            alpha *= (1 - 0.0003 * (n_searches - i) / n_searches)
        print(f"New alpha: {alpha}")
    assert len(point_poses) != 0, "No points found."
    return torch.tensor(point_poses, device=torch.device("cpu"),
                        dtype=torch.float, requires_grad=True), B


def get_experiments(alpha: float = 0.5):
    # update in time
    delta_t = 1.0

    # grid size
    N = 256

    # simulation steps
    N_simulation_steps = 10000

    experiments = {
        "base": {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
        },
        "more diffusion": {  # less points
            "DA": 0.16,
            "DB": 0.08,
            "f": 0.035,
            "k": 0.065,
        },
        "less diffusion": {  # more points
            "DA": 0.12,
            "DB": 0.04,
            "f": 0.035,
            "k": 0.065,
        },
    }

    # evaluate alpha
    assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"

    # mapping alpha param to DA and DB where alpha = 0.5 is the base case
    # and a higher alpha means more points
    experiment = experiments["base"]
    for key in ["DA", "DB"]:
        experiment[key] = experiment[key] - (alpha - 0.5) * .1

    return delta_t, experiment, N_simulation_steps, N
