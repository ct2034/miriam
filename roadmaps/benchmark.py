import abc
import timeit
from random import Random
from typing import Any, Dict, List, Optional, Tuple

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


class RoadmapToTest:
    def __init__(self, map_fname: str, rng: Random, roadmap_specific_kwargs: Dict[str, Any] = {}):
        self.map_fname = map_fname
        self.map_img = read_map(map_fname)
        self.n_eval = 100
        self.rng = rng
        self.roadmap_specific_kwargs = roadmap_specific_kwargs
        self.g: Optional[nx.Graph] = None
        self.runtime_ms: Optional[float] = None

    def _initialize_eval_rng(self):
        self.rng = Random(0)

    def evaluate_path_length(self) -> Dict[str, float]:
        assert self.g is not None, "Roadmap must be built."
        paths = make_paths(self.g, self.n_eval, self.map_img, self.rng)
        pos_t = torch.zeros((max(self.g.nodes) + 1, 2))
        for n, (x, y) in nx.get_node_attributes(self.g, POS).items():
            pos_t[n] = torch.tensor([x, y])
        lens = []
        for path in paths:
            try:
                lens.append(
                    get_path_len(pos_t, path, False)
                )
            except IndexError:
                pass
        return {"path_len": (sum(lens) / len(lens)).item(),
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


class Gsorm(RoadmapToTest):
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
        pos = torch.Tensor(nodes) / len(self.map_img)
        n = pos.shape[0]
        start_t = timeit.default_timer()
        self.g, _ = make_graph_and_flann(pos, self.map_img, n, rng)
        runtime_delaunay = (timeit.default_timer() - start_t) * 1000
        self.runtime_ms = runtime_points + runtime_delaunay


class Spars(RoadmapToTest):
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


def run():
    df = pd.DataFrame()

    trials = [
        (Gsorm, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 10000,
            "resolution": 300,
        }),
        (Gsorm, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 10000,
            "resolution": 400,
        }),
        (Gsorm, {
            "DA": 0.14,
            "DB": 0.06,
            "f": 0.035,
            "k": 0.065,
            "delta_t": 1.0,
            "iterations": 10000,
            "resolution": 500,
        }),
        (Spars, {
            "denseDelta":    10.,
            "sparseDelta":    100,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 1.,
        }),
        (Spars, {
            "denseDelta":    7.,
            "sparseDelta":    70.,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 1.,
        }),
        (Spars, {
            "denseDelta":    3.,
            "sparseDelta":    30.,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 1.,
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
    for cls, args in trials:
        for map_name in ["c", "x", "z"]:
            for seed in range(2):
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

    df.head()
    df.to_csv(CSV_PATH)


def plot():
    df = pd.read_csv(CSV_PATH)
    sns.set_theme(style="whitegrid")
    sns.pairplot(df, hue="roadmap", vars=[
                 "path_len", "n_nodes", "success_rate", "runtime_ms"])
    plt.savefig(CSV_PATH.replace(".csv", ".png"))


if __name__ == "__main__":
    run()
    plot()
