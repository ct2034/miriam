import abc
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

    def evaluate_path_length(self) -> Dict[str, float]:
        self.rng = Random(0)
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

    def evaluate(self):
        results = {}
        for fun in [
            self.evaluate_path_length,
            self.evaluate_n_nodes,
        ]:
            results.update(fun())
        return results


class Spars(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)

        from roadmaps.SPARS.build.libsparspy import Spars
        s = Spars()
        edges = s.run(
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

        pos = sample_points(n, self.map_img, self.rng)
        self.g, _ = make_graph_and_flann(pos, self.map_img, n, rng)

        optimizer = torch.optim.Adam([pos], lr=lr)

        for i_e in tqdm(range(epochs)):
            self.g, pos, test_length, training_length = optimize_poses(
                self.g, pos, self.map_img, optimizer, n, rng)


def run():
    df = pd.DataFrame()

    trials = [
        (Spars, {
            "denseDelta":    10.,
            "sparseDelta":    100,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 1.,
        }), (Spars, {
            "denseDelta":    7.,
            "sparseDelta":    70.,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 1.,
        }), (Spars, {
            "denseDelta":    3.,
            "sparseDelta":    30.,
            "stretchFactor": 1.01,
            "maxFailures": 500,
            "maxTime": 1.,
        }), (ODRM, {
            "n": 100,
            "lr": 1e-3,
            "epochs": 50,
        }), (ODRM, {
            "n": 300,
            "lr": 1e-3,
            "epochs": 50,
        }), (ODRM, {
            "n": 500,
            "lr": 1e-3,
            "epochs": 50,
        })
    ]
    for cls, args in trials:
        for map_name in ["c", "x", "z"]:
            i = len(df) + 1
            map_fname = f"roadmaps/odrm/odrm_eval/maps/{map_name}.png"
            t = cls(map_fname, Random(0), args)
            data = t.evaluate()
            for k, v in data.items():
                df.at[i, k] = v
            for k, v in args.items():
                df.at[i, cls.__name__ + "_" + k] = v
            df.at[i, "map"] = map_name
            df.at[i, "roadmap"] = cls.__name__

    df.head()
    df.to_csv(CSV_PATH)


def plot():
    df = pd.read_csv(CSV_PATH)
    sns.set_theme(style="whitegrid")
    sns.pairplot(df, hue="roadmap", vars=[
                 "path_len", "n_nodes", "success_rate"])
    plt.savefig(CSV_PATH.replace(".csv", ".png"))


if __name__ == "__main__":
    run()
    plot()
