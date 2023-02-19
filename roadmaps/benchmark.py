from itertools import product
import logging
from math import ceil, sqrt
import os
import re
import timeit
from random import Random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bresenham import bresenham
from matplotlib import pyplot as plt
from pyflann import FLANN
from tqdm import tqdm

from definitions import DISTANCE, POS
from roadmaps.var_odrm_torch.var_odrm_torch import (
    get_path_len, is_coord_free, is_pixel_free, make_paths,
    plan_path_between_coordinates, read_map)

CSV_PATH = "roadmaps/benchmark.csv"
PLOT_FOLDER = "roadmaps/benchmark_plots"
DPI = 500

logger = logging.getLogger(__name__)

# this list is roughly sorted by complexity
MAP_NAMES = [
    'plain',
    'c',
    'x',
    'b',
    'o',
    'dual_w',
    'dual2',
    'dual',
    'simple',
    'z'
]


class RoadmapToTest:
    def __init__(self, map_fname: str, rng: Random,
                 roadmap_specific_kwargs: Dict[str, Any] = {}):
        self.map_fname = map_fname
        self.map_img = read_map(map_fname)
        # swap rows and columns
        # self.map_img = np.swapaxes(np.array(self.map_img), 0, 1)
        self.n_eval = 100
        self.rng = rng
        self.roadmap_specific_kwargs = roadmap_specific_kwargs
        self.g: Optional[nx.Graph] = None
        self.runtime_ms: Optional[float] = None

    def _initialize_eval_rng(self):
        self.rng = Random(0)

    def evaluate_path_length(self) -> Dict[str, float]:
        assert self.g is not None, "Roadmap must be built."
        if self.g.number_of_nodes() == 0:
            logger.error("Error while making paths, no nodes in graph.")
            return {"success_rate": 0.0}
        map_img_inv = np.swapaxes(np.array(self.map_img), 0, 1)
        paths = make_paths(self.g, self.n_eval,
                           tuple(map_img_inv.tolist()),
                           self.rng)
        pos_t = torch.zeros((max(self.g.nodes) + 1, 2))
        for n, (x, y) in nx.get_node_attributes(self.g, POS).items():
            pos_t[n] = torch.tensor([x, y])
        lens: List[float] = []
        for path in paths:
            try:
                lens.append(
                    get_path_len(pos_t, path, False).item()
                )
            except IndexError:
                pass
        data = {"success_rate": len(lens) / self.n_eval}
        for i, d in enumerate(lens):
            data[f"path_length_{i:02d}"] = d
        data["path_length_mean"] = np.mean(lens).item()
        return data

    def _check_line(self, a: Tuple[float, float], b: Tuple[float, float]):
        line = bresenham(
            round(a[0] * len(self.map_img)),
            round(a[1] * len(self.map_img)),
            round(b[0] * len(self.map_img)),
            round(b[1] * len(self.map_img)),
        )
        # print(list(line))
        return all([is_pixel_free(
            self.map_img, (x[0], x[1])) for x in line])

    def evaluate_straight_path_length(self) -> Dict[str, float]:
        assert self.g is not None, "Roadmap must be built."
        if self.g.number_of_nodes() == 0:
            logger.error("Error while making paths, no nodes in graph.")
            return {"success_rate": 0.0}
        pos_np = np.zeros((max(self.g.nodes) + 1, 2))
        for n, (x, y) in nx.get_node_attributes(self.g, POS).items():
            pos_np[n] = torch.tensor([x, y])
        pos_t = torch.tensor(pos_np)
        flann = FLANN(random_seed=0)
        flann.build_index(np.array(pos_np, dtype=np.float32), random_seed=0)
        rel_straight_lengths: List[float] = []
        for _ in range(self.n_eval):
            found = False
            while not found:
                start = (self.rng.random(), self.rng.random())
                if not is_coord_free(self.map_img, start):
                    continue
                end = (self.rng.random(), self.rng.random())
                if not is_coord_free(self.map_img, end):
                    continue
                if not self._check_line(start, end):
                    continue
                length = np.linalg.norm(np.array(start) - np.array(end)).item()
                if length < 0.5:
                    continue
                try:
                    path = plan_path_between_coordinates(
                        self.g, flann, start, end)
                except Exception:
                    logger.error("Error while planning path.: ", exc_info=True)
                    continue
                if path is None:
                    continue
                found = True
            assert path is not None
            path_len = get_path_len(pos_t, path, False).item()
            rel_straight_lengths.append(path_len / length)
        data = {}
        for i, d in enumerate(rel_straight_lengths):
            data[f"rel_straight_length_{i:02d}"] = d
        data["rel_straight_length_mean"] = np.mean(rel_straight_lengths).item()
        return data

    def evaluate_n_nodes(self) -> Dict[str, float]:
        assert self.g is not None, "Roadmap must be built."
        return {"n_nodes": self.g.number_of_nodes()}

    def evaluate_runtime(self) -> Dict[str, float]:
        if self.runtime_ms is None:
            return {}
        return {"runtime_ms": self.runtime_ms}

    def evaluate(self):
        results = {}
        for fun in [
            self.evaluate_path_length,
            self.evaluate_straight_path_length,
            self.evaluate_n_nodes,
            self.evaluate_runtime,
        ]:
            self._initialize_eval_rng()
            results.update(fun())
        return results

    def plot(self, folder: str, i: int):
        assert self.g is not None, "Roadmap must be built."
        fig, ax = plt.subplots(dpi=DPI)
        ax.imshow(self.map_img, cmap="gray")

        # plot graph
        pos = nx.get_node_attributes(self.g, POS)
        pos = {k: (v[0] * len(self.map_img), v[1] * len(self.map_img))
               for k, v in pos.items()}
        nx.draw_networkx_nodes(self.g, pos, ax=ax, node_size=1)
        # exclude self edges
        edges = [(u, v) for u, v in self.g.edges if u != v]
        nx.draw_networkx_edges(self.g, pos, ax=ax, edgelist=edges, width=0.5)

        # what will be the file name
        n_nodes = self.g.number_of_nodes()
        if n_nodes == 0:
            return
        name = f"{i:03d}_{self.__class__.__name__}_" +\
            f"{n_nodes:04d}n_" +\
            f"{os.path.basename(self.map_fname)}"

        # make an example path
        # invert x and y of map
        map_img = np.swapaxes(np.array(self.map_img), 0, 1)
        map_img_t = tuple(map_img.tolist())
        seed = 0
        path = None
        while path is None:
            try:
                path = make_paths(self.g, 1, map_img_t, Random(seed))[0]
            except ValueError:
                seed += 1
        start, end, node_path = path
        coord_path = [pos[n] for n in node_path]
        full_path = (
            [(start[0] * len(self.map_img),
              start[1] * len(self.map_img))] +
            coord_path +
            [(end[0] * len(self.map_img),
              end[1] * len(self.map_img))])
        x, y = zip(*full_path)
        ax.plot(x, y, color="red", linewidth=0.5)

        fig.savefig(os.path.join(folder, name))
        plt.close(fig)


class GSRM(RoadmapToTest):
    """Its gsorm without the o.
    So no optimization, only the generation of the points by the
    Grey Scott model."""

    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        if "plot" not in roadmap_specific_kwargs:
            roadmap_specific_kwargs["plot"] = False

        from roadmaps.gsorm.build.libgsorm import Gsorm
        from roadmaps.var_odrm_torch.var_odrm_torch import make_graph_and_flann
        gs = Gsorm()
        nodes, runtime_points = gs.run(
            mapFile=map_fname,
            **self.roadmap_specific_kwargs,
        )
        pos = torch.Tensor(nodes) / roadmap_specific_kwargs["resolution"]
        # swap x and y
        pos = pos[:, [1, 0]]
        n = pos.shape[0]
        start_t = timeit.default_timer()
        self.g, _ = make_graph_and_flann(pos, self.map_img, n, rng)
        runtime_delaunay = (timeit.default_timer() - start_t) * 1000
        self.runtime_ms = runtime_points + runtime_delaunay

        # swap x and y
        nx.set_node_attributes(self.g,
                               {i: (p[1],
                                    p[0]) for i, p in nx.get_node_attributes(
                                   self.g, POS).items()}, POS)


class GSORM(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        from roadmaps.gsorm.build.libgsorm import Gsorm
        from roadmaps.var_odrm_torch.var_odrm_torch import (
            make_graph_and_flann, optimize_poses)

        # prepare args
        gsorm_kwargs = self.roadmap_specific_kwargs.copy()
        epochs_optim = gsorm_kwargs.pop("epochs_optim", None)
        lr_optim = gsorm_kwargs.pop("lr_optim", None)

        # run gsorm
        gs = Gsorm()
        nodes, runtime_points = gs.run(
            mapFile=map_fname,
            **gsorm_kwargs,
            plot=False,
        )
        pos = torch.Tensor(nodes) / roadmap_specific_kwargs["resolution"]
        # swap x and y
        pos = pos[:, [1, 0]]
        pos.requires_grad = True
        n = pos.shape[0]

        # optimize
        optimizer = torch.optim.Adam([pos], lr=lr_optim)
        start_t = timeit.default_timer()
        self.g, _ = make_graph_and_flann(pos, self.map_img, n, rng)
        for i_e in tqdm(range(epochs_optim)):
            self.g, pos, test_length, training_length = optimize_poses(
                self.g, pos, self.map_img, optimizer, n, rng)
        runtime_optim = (timeit.default_timer() - start_t) * 1000
        self.runtime_ms = runtime_points + runtime_optim

        # swap x and y
        nx.set_node_attributes(self.g,
                               {i: (p[1],
                                    p[0]) for i, p in nx.get_node_attributes(
                                   self.g, POS).items()}, POS)


class SPARS(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)

        from roadmaps.SPARS.build.libsparspy import Spars
        s = Spars()
        edges, self.runtime_ms = s.run(
            mapFile=map_fname,
            seed=rng.randint(0, 2**16),
            **self.roadmap_specific_kwargs,
        )

        # to networkx
        self.g = nx.Graph()
        for (a, ax, ay, b, bx, by) in edges:
            # positions to unit square
            ax /= len(self.map_img)
            ay /= len(self.map_img)
            bx /= len(self.map_img)
            by /= len(self.map_img)
            self.g.add_node(a, **{POS: (ax, ay)})
            self.g.add_node(b, **{POS: (bx, by)})
            self.g.add_edge(
                a, b, **{DISTANCE: np.sqrt((ax - bx)**2 + (ay - by)**2)})
        print("nodes", self.g.number_of_nodes())
        print("edges", self.g.number_of_edges())

        # plot
        # fig, ax = plt.subplots(figsize=(10, 10), dpi=DPI)
        # nx.draw(
        #     self.g,
        #     ax=ax,
        #     node_size=4,
        #     width=0.2,
        #     pos=nx.get_node_attributes(self.g, POS),
        # )
        # ax.set_aspect("equal")
        # fig.tight_layout()
        # fig.savefig("a.png")


class ODRM(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        from roadmaps.var_odrm_torch.var_odrm_torch import (
            make_graph_and_flann, optimize_poses, sample_points)
        n = roadmap_specific_kwargs["n"]
        epochs = roadmap_specific_kwargs["epochs"]
        lr = roadmap_specific_kwargs["lr"]

        start_t = timeit.default_timer()
        pos = sample_points(n, self.map_img, self.rng)
        self.g, _ = make_graph_and_flann(pos, self.map_img, n, rng)

        optimizer = torch.optim.Adam([pos], lr=lr)

        for i_e in tqdm(range(epochs)):
            self.g, pos, test_length, training_length = optimize_poses(
                self.g, pos, self.map_img, optimizer, n, rng)
        end_t = timeit.default_timer()
        self.runtime_ms = (end_t - start_t) * 1000

        # swap x and y
        nx.set_node_attributes(self.g,
                               {i: (p[1],
                                    p[0]) for i, p in nx.get_node_attributes(
                                   self.g, POS).items()}, POS)


class PRM(RoadmapToTest):
    # Hint: PRM is a special case of ODRM without optimization
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        from roadmaps.var_odrm_torch.var_odrm_torch import (
            make_graph_and_flann, sample_points)
        n = roadmap_specific_kwargs["n"]

        start_t = timeit.default_timer()
        pos = sample_points(n, self.map_img, self.rng)
        self.g, _ = make_graph_and_flann(pos, self.map_img, n, rng)

        end_t = timeit.default_timer()
        self.runtime_ms = (end_t - start_t) * 1000

        # swap x and y
        nx.set_node_attributes(self.g,
                               {i: (p[1],
                                    p[0]) for i, p in nx.get_node_attributes(
                                   self.g, POS).items()}, POS)


class GridMap(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        n_side = ceil(sqrt(roadmap_specific_kwargs["n"]))
        start_t = timeit.default_timer()
        g = self._make_gridmap(n_side)
        while g.number_of_nodes() < roadmap_specific_kwargs["n"]:
            n_side += 1
            g = self._make_gridmap(n_side)
        end_t = timeit.default_timer()
        self.runtime_ms = (end_t - start_t) * 1000
        self.g = g

    def _make_gridmap(self, n_side):
        edge_length = 1 / (n_side + 1)
        g = nx.Graph()
        grid = np.full((n_side, n_side), -1)
        for x, y in product(range(n_side), range(n_side)):
            i_to_add = len(g)
            coords = (
                x * edge_length + edge_length / 2,
                y * edge_length + edge_length / 2,
            )
            if is_coord_free(self.map_img, coords):
                g.add_node(i_to_add, **{POS: coords})
                grid[x, y] = i_to_add
                if x > 0 and grid[x - 1, y] != -1:
                    if self._check_line(
                            coords,
                            g.nodes[grid[x - 1, y]][POS]):
                        g.add_edge(i_to_add, grid[x - 1, y], **{
                            DISTANCE: edge_length
                        })
                if y > 0 and grid[x, y - 1] != -1:
                    if self._check_line(
                            coords,
                            g.nodes[grid[x, y - 1]][POS]):
                        g.add_edge(i_to_add, grid[x, y - 1], **{
                            DISTANCE: edge_length
                        })
        # swap x and y
        nx.set_node_attributes(
            g,
            {i: (p[1],
                 p[0]) for i, p in nx.get_node_attributes(
                g, POS).items()}, POS)
        return g


def run():
    df = pd.DataFrame()

    trials = [
        (GSRM, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 10000,
            "resolution": 300,
            "plot": True,
        }),
        (GSRM, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 10000,
            "resolution": 400,
        }),
        (GSRM, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 10000,
            "resolution": 500,
        }),
        (GSORM, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 5000,  # of grey scott model
            "resolution": 300,
            "epochs_optim": 25,  # of optimization
            "lr_optim": 1e-3,
        }),
        (GSORM, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 5000,
            "resolution": 400,
            "epochs_optim": 25,  # of optimization
            "lr_optim": 1e-3,
        }),
        (GSORM, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 5000,
            "resolution": 500,
            "epochs_optim": 25,  # of optimization
            "lr_optim": 1e-3,
        }),
        (SPARS, {
            "denseDelta":    80.,
            "sparseDelta":    800,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 8.,
        }),
        (SPARS, {
            "denseDelta":    50.,
            "sparseDelta":    500.,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 8.,
        }),
        (SPARS, {
            "denseDelta":    20.,
            "sparseDelta":    200.,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 8.,
        }),
        (ODRM, {
            "n": 400,
            "lr": 1e-3,
            "epochs": 50,
        }),
        (ODRM, {
            "n": 900,
            "lr": 1e-3,
            "epochs": 50,
        }),
        (ODRM, {
            "n": 1400,
            "lr": 1e-3,
            "epochs": 50,
        }),
        (PRM, {
            "n": 500,
        }),
        (PRM, {
            "n": 900,
        }),
        (PRM, {
            "n": 1500,
        }),
        (GridMap, {
            "n": 400,
        }),
        (GridMap, {
            "n": 900,
        }),
        (GridMap, {
            "n": 1400,
        })
        # TODO: visibility graphs, voronoi diagrams
    ]
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)
    if not len(os.listdir(PLOT_FOLDER)) == 0:
        # delete all files in folder
        for f in os.listdir(PLOT_FOLDER):
            os.remove(os.path.join(PLOT_FOLDER, f))
    params_to_run = []
    df["i"] = np.nan
    df.set_index("i", inplace=True)
    for cls, args in trials:
        for map_name in map_names:
            if "plot" in args:
                if map_name == "c" and args["plot"]:
                    print(f"Plotting {len(df) + 1}")
                else:
                    args["plot"] = False

            for seed in range(5):
                i = len(df) + 1  # new experiment in new row
                df.at[i, "i"] = i
                df.at[i, "map"] = map_name
                df.at[i, "roadmap"] = cls.__name__
                df.at[i, "seed"] = seed
                params_to_run.append((cls, args, map_name, seed, i))
    Random(0).shuffle(params_to_run)
    df = df.copy()

    for ptr in tqdm(params_to_run):
        cls, args, map_name, seed, i = ptr
        _, data = _run_proxy(ptr)
        for k, v in data.items():
            df.at[i, k] = v
        for k, v in args.items():
            df.at[i, cls.__name__ + "_" + k] = v
    # with Pool(2) as p:
    #     for i, data in p.imap_unordered(_run_proxy, params_to_run):
    #         for k, v in data.items():
    #             df.at[i, k] = v
    #         for k, v in args.items():
    #             df.at[i, cls.__name__ + "_" + k] = v

    df.head()
    df.to_csv(CSV_PATH)


def _run_proxy(args):
    cls, args, map_name, seed, i = args
    map_fname = f"roadmaps/odrm/odrm_eval/maps/{map_name}.png"
    t = cls(map_fname, Random(seed), args)
    data = t.evaluate()
    t.plot(PLOT_FOLDER, i)
    return (i, data)


def _get_i_from_column_name(col_name):
    return int(col_name.split("_")[-1])


def _get_cols_by_prefix(df, prefix):
    return filter(
        # must math {prefix}XX
        lambda x: re.match(f"^({prefix})[0-9]+$", x),
        df.columns)


def plot():
    interesting_vars = [
        "path_length_mean",
        "rel_straight_length_mean",
        "n_nodes",
        "success_rate",
        "runtime_ms"
    ]

    df = pd.read_csv(CSV_PATH)
    sns.set_theme(style="whitegrid")
    sns.set(rc={'figure.dpi': DPI})
    sns.pairplot(
        df,
        hue="roadmap",
        vars=interesting_vars,
        markers=".",
    )
    plt.savefig(CSV_PATH.replace(".csv", ".png"))
    plt.close('all')

    for roadmap in df.roadmap.unique():
        df_roadmap = df[df.roadmap == roadmap]
        sns.set_theme(style="whitegrid")
        sns.pairplot(
            df_roadmap,
            hue="map",
            x_vars=[
                col for col in df_roadmap.columns if col.startswith(roadmap)],
            y_vars=interesting_vars,
        )
        plt.savefig(CSV_PATH.replace(".csv", f"_{roadmap}.png"))
        plt.close('all')

    COMPARE_TO = "GSORM"
    for roadmap in df.roadmap.unique():
        if roadmap == COMPARE_TO:
            continue
        df_roadmap = df[df.roadmap == roadmap]
        df_compare = df[df.roadmap == COMPARE_TO]
        data_len = []
        data_rel = []
        map_names = []
        assert len(df_roadmap) == len(df_compare), \
            "df_roadmap and df_compare must have same length"
        for data, prefix in [
            (data_len, "path_length_"),
            (data_rel, "rel_straight_length_")
        ]:
            cols = _get_cols_by_prefix(df_roadmap, prefix)
            i_s = [_get_i_from_column_name(col) for col in cols]
            map_names = []
            for row in range(len(df_roadmap)):
                map_name = df_roadmap.iloc[row]["map"]
                for i in i_s:
                    data.append((
                        df_compare.iloc[row][f"{prefix}{i:02d}"],
                        df_roadmap.iloc[row][f"{prefix}{i:02d}"],
                    ))
                    map_names.append(map_name)
        assert len(data_len) == len(data_rel), \
            "data_len and data_rel must have same length"
        assert len(data_len) == len(map_names), \
            "data_len and map_names must have same length"
        data_len_np = np.array(data_len)
        data_rel_np = np.array(data_rel)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=DPI)
        # sort by map names list
        sorted_map_names = []
        for map_name in MAP_NAMES:
            if map_name in list(set(map_names)):
                sorted_map_names.append(map_name)
        colors_per_data = [
            plt.cm.get_cmap("hsv")(
                sorted_map_names.index(map_name) /
                len(sorted_map_names))
            for map_name in map_names
        ]
        for i, d in enumerate([data_len_np, data_rel_np]):
            axs[i].scatter(d[:, 0],
                           d[:, 1],
                           marker='.',
                           alpha=.4,
                           c=colors_per_data,
                           edgecolors='none',
                           s=2)
            axs[i].set_xlabel(COMPARE_TO)
            axs[i].set_ylabel(roadmap)
            axs[i].plot([0, 1000], [0, 1000], color="black", linewidth=1)
            axs[i].set_xlim(min(d[:, 0]) - .1, max(d[:, 0]) + .1)
            axs[i].set_ylim(min(d[:, 1]) - .1, max(d[:, 1]) + .1)
        axs[0].set_title("Path Length")
        axs[1].set_title("Relative Path Length")

        # make legend
        legend_elements = [
            plt.Line2D([-1], [-1],
                       marker='.',
                       color='w',
                       label=map_name,
                       markerfacecolor=plt.cm.get_cmap("hsv")(
                       sorted_map_names.index(map_name) /
                       len(sorted_map_names)),
                       markersize=10)
            for map_name in sorted_map_names
        ]
        axs[0].legend(handles=legend_elements, loc='upper left')

        fig.savefig(CSV_PATH.replace(".csv", f"_{roadmap}_scatter.png"))
        plt.close('all')


if __name__ == "__main__":
    # run()
    plot()
