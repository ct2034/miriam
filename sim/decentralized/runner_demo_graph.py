import logging
import random
from typing import Any, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from definitions import POS
from libpysal import weights
from libpysal.cg import voronoi_frames
from scenarios.visualization import plot_env_with_arrows, plot_with_paths
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import OptimalPolicy
from sim.decentralized.runner import run_a_scenario, will_agents_collide

if __name__ == "__main__":
    logging.getLogger("sim.decentralized.runner").setLevel(logging.DEBUG)
    logging.getLogger("sim.decentralized.iterators").setLevel(logging.DEBUG)

    # params
    n_agents = 8
    n_nodes = 32
    rng = random.Random(0)

    # random points for node positions
    x = [rng.uniform(0, 1) for _ in range(n_nodes)]
    y = [rng.uniform(0, 1) for _ in range(n_nodes)]
    pos_s = list(zip(x, y))
    pos_s_np = np.array(list(pos_s))
    pos_s_dict = {i: pos_s[i] for i in range(n_nodes)}

    # make delaunay graph from positions
    cells, generators = voronoi_frames(pos_s_np, clip="convex hull")
    delaunay = weights.Rook.from_dataframe(cells)
    g = delaunay.to_networkx()
    nx.set_node_attributes(g, pos_s_dict, POS)

    # into a scenario
    starts = rng.sample(g.nodes(), n_agents)
    goals = rng.sample(g.nodes(), n_agents)
    plot_env_with_arrows(g, starts, goals)

    # initialize agents
    agents = tuple([Agent(g, start, radius=0.01) for start in starts])
    for i, agent in enumerate(agents):
        agent.give_a_goal(goals[i])
        agent.policy = OptimalPolicy(agent, None)

    # independent paths
    do_collide, paths = will_agents_collide(agents, ignore_finished_agents=True)
    print(f"Will agents collide? {do_collide}")
    plot_with_paths(g, paths)

    # run the scenario
    paths_run: List[Any] = []
    res = run_a_scenario(
        env=g,
        agents=agents,
        plot=False,
        iterator=IteratorType.LOOKAHEAD2,
        paths_out=paths_run,
    )
    print(f"{res=}")

    plot_with_paths(g, paths_run)

    plt.show()
