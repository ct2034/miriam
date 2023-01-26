import abc
from random import Random
from typing import Dict, List, Tuple

import networkx as nx
from matplotlib import pyplot as plt

from roadmaps.var_odrm_torch.var_odrm_torch import read_map


class RoadmapToTest:
    @abc.abstractmethod
    def __init__(self, map_fname: str, rng: Random):
        pass

    @abc.abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass


class Spars:
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        from roadmaps.SPARS.build.libsparspy import Spars
        s = Spars()
        edges = s.run(
            mapFile=map_fname,
            seed=rng.randint(0, 2**16),
            **roadmap_specific_kwargs,
        )

        # to networkx
        self.rng = rng
        self.g = nx.Graph()
        for (a, ax, ay, b, bx, by) in edges:
            self.g.add_node(a, pos=(ax, ay))
            self.g.add_node(b, pos=(bx, by))
            self.g.add_edge(a, b)
        print("nodes", self.g.number_of_nodes())
        print("edges", self.g.number_of_edges())

        # plot
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        nx.draw(
            self.g,
            ax=ax,
            node_size=4,
            width=0.2,
            pos=nx.get_node_attributes(self.g, "pos"),
        )
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig("a.png")

    def evaluate(self) -> Dict[str, float]:
        return {"time": 1, "path": 2}


if __name__ == "__main__":
    sparse_specific_kwargs = {
        "denseDelta":    1.,
        "sparseDelta":    10.,
        "stretchFactor": 1.01,
        "maxFailures": 500,
        "maxTime": 1.,
    }
    rm = Spars("roadmaps/odrm/odrm_eval/maps/x.png",
               Random(0), sparse_specific_kwargs)
    rm.evaluate()
