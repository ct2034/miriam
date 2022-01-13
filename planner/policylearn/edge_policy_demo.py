import networkx as nx
from definitions import POS
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy_graph_utils import agents_to_data
from planner.policylearn.generate_data_demo import plot_graph
from sim.decentralized.agent import Agent

if __name__ == "__main__":
    g = nx.Graph()
    g.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
    ])
    nx.set_node_attributes(g, {
        0: (0., 0.),
        1: (1., 0.),
        2: (1., 1.),
        3: (.5, 1.7),
        4: (0., 1.)}, POS)

    agents = [
        Agent(g, 0, radius=.1),
        Agent(g, 4, radius=.1)
    ]
    agents[0].give_a_goal(3)
    agents[1].give_a_goal(1)

    data, _, _ = agents_to_data(agents, 0)

    f, ax = plt.subplots(1, 1)
    plot_graph(ax, data.edge_index, data.pos, data.x)
    plt.show()
