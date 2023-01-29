import os
import timeit
from random import Random
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from definitions import DISTANCE, POS
from roadmaps.var_odrm_torch.var_odrm_torch import (get_path_len, make_paths,
                                                    read_map)

CSV_PATH = "roadmaps/benchmark.csv"
PLOT_FOLDER = "roadmaps/benchmark_plots"


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
        map_img_inv = np.swapaxes(np.array(self.map_img), 0, 1)
        paths = make_paths(self.g, self.n_eval,
                           tuple(map_img_inv.tolist()),
                           self.rng)
        pos_t = torch.zeros((max(self.g.nodes) + 1, 2))
        for n, (x, y) in nx.get_node_attributes(self.g, POS).items():
            pos_t[n] = torch.tensor([x, y])
        lens = []
        for path in paths:
            try:
                lens.append(
                    get_path_len(pos_t, path, False).item()
                )
            except IndexError:
                pass
        return {"path_len": (sum(lens) / len(lens)),
                "success_rate": len(lens) / self.n_eval}

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
            self.evaluate_n_nodes,
            self.evaluate_runtime,
        ]:
            self._initialize_eval_rng()
            results.update(fun())
        return results

    def plot(self, folder: str, i: int):
        assert self.g is not None, "Roadmap must be built."
        fig, ax = plt.subplots(dpi=500)
        ax.imshow(self.map_img, cmap="gray")

        # plot graph
        pos = nx.get_node_attributes(self.g, POS)
        pos = {k: (v[0] * len(self.map_img), v[1] * len(self.map_img))
               for k, v in pos.items()}
        nx.draw_networkx_nodes(self.g, pos, ax=ax, node_size=1)
        # exclude self edges
        edges = [(u, v) for u, v in self.g.edges if u != v]
        nx.draw_networkx_edges(self.g, pos, ax=ax, edgelist=edges, width=0.5)
        name = f"{i:03d}_{self.__class__.__name__}_" +\
            f"{os.path.basename(self.map_fname)}"

        # make an example path
        # invert x and y of map
        map_img = np.swapaxes(np.array(self.map_img), 0, 1)
        map_img_t = tuple(map_img.tolist())
        path = make_paths(self.g, 1, map_img_t, Random(0))[0]
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


class GSRM(RoadmapToTest):
    """Its gsorm without the o.
    So no optimization, only the generation of the points by the
    Grey Scott model."""

    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)

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
        # fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
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


def run():
    df = pd.DataFrame()

    trials = [
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
        (GSRM, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 10000,
            "resolution": 300,
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
        (SPARS, {
            "denseDelta":    10.,
            "sparseDelta":    100,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 5.,
        }),
        (SPARS, {
            "denseDelta":    7.,
            "sparseDelta":    70.,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 5.,
        }),
        (SPARS, {
            "denseDelta":    3.,
            "sparseDelta":    30.,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 5.,
        }),
        (ODRM, {
            "n": 100,
            "lr": 1e-3,
            "epochs": 50,
        }),
        (ODRM, {
            "n": 300,
            "lr": 1e-3,
            "epochs": 50,
        }),
        (ODRM, {
            "n": 500,
            "lr": 1e-3,
            "epochs": 50,
        })
    ]
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)
    if not len(os.listdir(PLOT_FOLDER)) == 0:
        # delete all files in folder
        for f in os.listdir(PLOT_FOLDER):
            os.remove(os.path.join(PLOT_FOLDER, f))
    for cls, args in trials:
        for map_name in [
            "b",
            "c",
            "dual_w",
            "dual",
            "dual2",
            # "o",
            # "plain",
            # "simple",
            # "x",
            # "z"
        ]:
            for seed in range(1):
                i = len(df) + 1  # new experiment in new row
                map_fname = f"roadmaps/odrm/odrm_eval/maps/{map_name}.png"
                df.at[i, "map"] = map_name
                t = cls(map_fname, Random(seed), args)
                df.at[i, "roadmap"] = cls.__name__
                df.at[i, "seed"] = seed
                data = t.evaluate()
                for k, v in data.items():
                    df.at[i, k] = v
                for k, v in args.items():
                    df.at[i, cls.__name__ + "_" + k] = v
                t.plot(PLOT_FOLDER, i)

    df.head()
    df.to_csv(CSV_PATH)


def plot():
    df = pd.read_csv(CSV_PATH)
    sns.set_theme(style="whitegrid")
    sns.pairplot(
        df,
        hue="roadmap",
        vars=[
            "path_len",
            "n_nodes",
            "success_rate",
            "runtime_ms"
        ],
        # markers=".",
    )
    plt.savefig(CSV_PATH.replace(".csv", ".png"))

    for roadmap in df.roadmap.unique():
        df_roadmap = df[df.roadmap == roadmap]
        sns.set_theme(style="whitegrid")
        sns.pairplot(
            df_roadmap,
            hue="map",
            x_vars=[
                col for col in df_roadmap.columns if col.startswith(roadmap)],
            y_vars=[
                "path_len",
                "n_nodes",
                "success_rate",
                "runtime_ms"
            ],
        )
        plt.savefig(CSV_PATH.replace(".csv", f"_{roadmap}.png"))


if __name__ == "__main__":
    run()
    plot()
