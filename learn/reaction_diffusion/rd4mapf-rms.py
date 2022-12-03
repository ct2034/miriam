#!/usr/bin/python3
import pickle as pkl
from itertools import product
from typing import Dict

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import laplace
from sklearn.cluster import *
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm

from definitions import POS
from roadmaps.var_odrm_torch.var_odrm_torch import make_graph_and_flann

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


def sim(gray_scott_update, get_initial_configuration, delta_t, N,
        N_simulation_steps, A_bg, B_bg, experiments, mask):
    results: Dict[str, Dict[str, np.ndarray]] = {}

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


def processing(mask):
    d = pkl.load(open("learn/reaction_diffusion/rd4mapf-rms.pkl", "rb"))
    ncols = len(d)
    fig, axes = plt.subplots(nrows=2, ncols=ncols)
    map_img = tuple((np.logical_not(mask) * 255).astype(np.uint8))
    for i, (experiment, data) in enumerate(d.items()):
        assert "B" in data.keys()
        bitmap = data["B"] > data["B"].mean() * 1.7
        pos = axes[0, i].imshow(bitmap, cmap="gray")
        axes[0, i].set_title(f"{experiment} bitmap")
        axes[0, i].axis("off")
        point_poses = bitmap_to_point_poses(bitmap)
        axes[0, i].scatter(
            point_poses[:, 1] * bitmap.shape[0],
            point_poses[:, 0] * bitmap.shape[0],
            s=1,
            c="red",
            marker=".",
            alpha=0.9)

        axes[1, i].set_title(f"{len(point_poses)} points ...")
        g, _ = make_graph_and_flann(pos=torch.Tensor(point_poses),
                                    map_img=map_img,
                                    desired_n_nodes=len(point_poses))
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
        pos_dict = {i: [pos[i][1], -pos[i][0]] for i in g.nodes()}
        for e in g.edges():
            a, b = e
            if a == b:
                g.remove_edge(a, b)
        nx.draw_networkx(g, pos_dict, **options)
        axes[1, i].set_aspect("equal")
        axes[1, i].axis("off")

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

    mask = np.zeros((N, N), dtype=bool)
    mask[N//3:N//3*2, 0:N//6] = True
    mask[N//6*1:N//6*5, N//6*4:N//6*5] = True

    # sim(gray_scott_update, get_initial_configuration,
    #     delta_t, N, N_simulation_steps, A_bg, B_bg, experiments, mask)
    draw()
    processing(mask)
