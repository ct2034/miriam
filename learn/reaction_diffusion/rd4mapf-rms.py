#!/usr/bin/python3
import pickle as pkl
from itertools import product
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import torch
from bresenham import bresenham
from matplotlib import pyplot as plt
from scipy.ndimage import laplace
from sklearn.cluster import *  # scikit-learn
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm

from definitions import POS
from roadmaps.var_odrm_torch.var_odrm_torch import make_graph_and_flann
from scenarios.generators import movingai_read_mapfile

# src: https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb


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


def get_initial_configuration(N, random_influence=0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize a concentration configuration. `N` is the side length
    of the (`N` x `N`)-sized grid.
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


def sim(delta_t, N,
        N_simulation_steps, A_bg, B_bg, experiments, mask):
    """Run the simulations based on the params defined in `experiments`."""
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for experiment_name, experiment in experiments.items():
        A, B = get_initial_configuration(N)
        for i in tqdm(range(N_simulation_steps)):
            A, B = gray_scott_update(
                A, B, A_bg, B_bg, mask, **experiment, delta_t=delta_t)
        results[experiment_name] = {"A": A, "B": B}
    pkl.dump(results, open("learn/reaction_diffusion/rd4mapf-rms.pkl", "wb"))


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
    fig.tight_lfile: strayout()
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


def processing_back_and_forth(mask, experiments):
    """Experiment to turn reaction diffusion into poses and back again."""
    d = pkl.load(open("learn/reaction_diffusion/rd4mapf-rms.pkl", "rb"))
    ncols = len(d)
    fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=(ncols*5, 4*5))
    map_img = tuple((np.logical_not(mask) * 255).astype(np.uint8))
    results: Dict[str, Dict[str, np.ndarray]] = {}
    poses_results: Dict[str, np.ndarray] = {}
    for i, (experiment, data) in enumerate(d.items()):
        assert "A" in data.keys()
        assert "B" in data.keys()
        A = data["A"]
        B = data["B"]
        results[experiment] = {"A0": A, "B0": B}

        A_min_max = (A.min(), A.max())
        B_min_max = (B.min(), B.max())
        A_bg = A_min_max[1]
        B_bg = B_min_max[0]

        # raction diffusion to poses
        bitmap = reaction_difussion_to_bitmap(B)
        point_poses = bitmap_to_point_poses(bitmap)
        poses_results[experiment] = point_poses
        radius = find_radius(point_poses, bitmap)

        # display poses
        plot_bitmap_and_poses(axes[0, i], bitmap,
                              point_poses, title=experiment)

        # make a graph from that
        g, _ = make_graph_and_flann(pos=torch.Tensor(point_poses),
                                    map_img=map_img,
                                    desired_n_nodes=len(point_poses))

        # display graph
        axes[1, i].set_title(f"{len(point_poses)} points ...")
        axes[1, i].imshow(np.logical_not(mask), cmap="gray", alpha=0.5)
        pos = nx.get_node_attributes(g, POS)
        options = {
            "ax": axes[1, i],
            "node_size": 20,
            "node_color": "red",
            "edgecolors": "grey",
            "linewidths": 0.1,
            "width": 1,
            "with_labels": False
        }
        pos_dict = {i: [pos[i][1]*mask.shape[0],
                        pos[i][0]*mask.shape[0]] for i in g.nodes()}
        for e in g.edges():
            a, b = e
            if a == b:
                g.remove_edge(a, b)
        nx.draw_networkx(g, pos_dict, **options)
        axes[1, i].set_aspect("equal")
        axes[1, i].axis("off")

        # make poses into reaction diffusion
        A, B = poses_to_reaction_diffusion(
            point_poses, A_min_max, B_min_max, bitmap.shape[0], radius)
        bitmap = reaction_difussion_to_bitmap(B)
        point_poses = bitmap_to_point_poses(bitmap)
        plot_bitmap_and_poses(axes[2, i], bitmap,
                              point_poses, title=experiment)
        results[experiment]["A1"] = A.copy()
        results[experiment]["B1"] = B.copy()

        # make poses into raction diffusion
        for _ in range(100):
            A, B = gray_scott_update(
                A, B, A_bg, B_bg, mask, **experiments[experiment], delta_t=1)
        bitmap = reaction_difussion_to_bitmap(B)
        point_poses = bitmap_to_point_poses(bitmap)
        plot_bitmap_and_poses(axes[3, i], bitmap,
                              point_poses, title=experiment)
        results[experiment]["A2"] = A.copy()
        results[experiment]["B2"] = B.copy()

    pkl.dump(results, open(
        "learn/reaction_diffusion/rd4mapf-rms-processed.pkl", "wb"))
    pkl.dump(poses_results, open(
        "learn/reaction_diffusion/rd4mapf-rms-poses.pkl", "wb"))
    fig.tight_layout()
    plt.savefig("learn/reaction_diffusion/rd4mapf-rms-poses.png", dpi=300)


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
    N = 256

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

    map_np = movingai_read_mapfile(
        "roadmaps/odrm/odrm_eval/maps/random-32-32-10.map").astype(np.bool8)
    rep = N // map_np.shape[0]
    mask = np.repeat(map_np, rep, axis=0).repeat(rep, axis=1)
    assert mask.shape == (N, N)

    sim(delta_t, N, N_simulation_steps, A_bg, B_bg, experiments, mask)
    draw("learn/reaction_diffusion/rd4mapf-rms")
    processing_back_and_forth(mask, experiments)
    draw("learn/reaction_diffusion/rd4mapf-rms-processed")
