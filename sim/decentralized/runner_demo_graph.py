import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from libpysal import examples, weights
from libpysal.cg import voronoi_frames
from scenarios.visualization import plot_env_with_arrows
from sim.decentralized.runner import initialize_new_agent

if __name__ == '__main__':
    g = nx.random_geometric_graph(100, 0.2, seed=0)
    pos = nx.get_node_attributes(g, 'pos')

    plt.figure(figsize=(10, 10))
    nx.draw(g, pos, node_size=10, node_color='blue', with_labels=False)

    # make delaunay graph from same poses
    pos_np = np.array(list(pos.values()))
    cells, generators = voronoi_frames(pos_np, clip="convex hull")
    delaunay = weights.Rook.from_dataframe(cells)
    g_del = delaunay.to_networkx()

    plt.figure(figsize=(10, 10))
    nx.draw(g_del, pos, node_size=10,
            node_color='blue', with_labels=False)

    plt.show()
