import logging
from random import Random

from definitions import INVALID
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr
from roadmaps.var_odrm_torch.var_odrm_torch import (make_graph_and_flann,
                                                    read_map, sample_points)

if __name__ == "__main__":
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.DEBUG)
    rng = Random(0)
    n_nodes = 32
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/z.png"
    map_img = read_map(map_fname)
    pos = sample_points(n_nodes, map_img, rng)
    g, _ = make_graph_and_flann(pos, map_img)
    n_agents = 3
    n_demo = 10
    success_rate = 0.0
    for i_demo in range(n_demo):
        starts = rng.sample(list(g.nodes()), k=n_agents)
        goals = rng.sample(list(g.nodes()), k=n_agents)
        paths = plan_cbsr(g, starts, goals, .001, 60, skip_cache=True)
        if paths is not INVALID:
            success_rate += (1.0 / n_demo)
    print(f"success rate: {success_rate:.1f}")
