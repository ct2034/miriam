import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from libpysal import examples, weights
from libpysal.cg import voronoi_frames
from scenarios.visualization import plot_env_with_arrows, plot_with_paths
from sim.decentralized.agent import Agent

if __name__ == '__main__':
    n_agents = 8
    n_nodes = 64
    rng = random.Random(0)

    # random points for node positions
    x = [rng.uniform(0, 1) for _ in range(n_nodes)]
    y = [rng.uniform(0, 1) for _ in range(n_nodes)]
    pos = list(zip(x, y))
    pos_np = np.array(list(pos))
    pos_dict = {i: pos[i] for i in range(n_nodes)}

    # make delaunay graph from positions
    cells, generators = voronoi_frames(pos_np, clip="convex hull")
    delaunay = weights.Rook.from_dataframe(cells)
    g = delaunay.to_networkx()
    nx.set_node_attributes(g, pos_dict, 'pos')

    # into a scenario
    starts = rng.sample(g.nodes(), n_agents)
    goals = rng.sample(g.nodes(), n_agents)
    plot_env_with_arrows(g, starts, goals)

    # initialize agents
    agents = [Agent(g, start) for start in starts]
    for i, agent in enumerate(agents):
        agent.give_a_goal(goals[i])

    # visualize paths
    paths = [agent.path for agent in agents]
    plot_with_paths(g, paths)

    plt.show()
