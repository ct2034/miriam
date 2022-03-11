from random import Random

import networkx as nx
import numpy as np
from definitions import INVALID, POS
from learn.delaunay_benchmark.delaunay_implementations import (
    read_map, run_delaunay_libpysal)
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr

if __name__ == "__main__":
    rng = Random(0)
    n_nodes = 250
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/z.png"
    map_img = read_map(map_fname)
    pos = np.array(
        [(rng.randint(0, map_img.shape[0]-1),
            rng.randint(0, map_img.shape[1]-1))
            for _ in range(n_nodes)],
        dtype=np.int32)
    g = run_delaunay_libpysal(pos, map_img)
    n_agents = 50
    starts = rng.sample(list(g.nodes()), k=n_agents)
    goals = rng.sample(list(g.nodes()), k=n_agents)
    paths = plan_cbsr(g, starts, goals, .005, 60, skip_cache=True)
    print(paths)
