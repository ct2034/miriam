import networkx as nx
from definitions import POS
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy_graph_utils import agents_to_data
from planner.policylearn.generate_data_demo import plot_graph_wo_pos_data
from sim.decentralized.agent import Agent

if __name__ == "__main__":
    g = nx.Graph()
    g.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (1, 5),
        (3, 7),
        (5, 6),
        (6, 7),
        (3, 8),
        (7, 8),
    ])
    nx.set_node_attributes(g, {
        0: (0., 0.),
        1: (1., 0.),
        2: (1., 1.),
        3: (.5, 1.7),
        4: (0., 1.),
        5: (2., 0.),
        6: (2., 1.),
        7: (1.5, 1.7),
        8: (1, 3.4),
    }, POS)

    agents = [
        Agent(g, 0, radius=.1),
        Agent(g, 3, radius=.1)
    ]
    agents[0].give_a_goal(5)
    agents[1].give_a_goal(0)

    # each make first step
    # agents[0].make_next_step(agents[0].path[1][0])
    # agents[1].make_next_step(agents[1].path[1][0])

    data, _, big_from_small = agents_to_data(
        agents=agents,
        i_self=0,
        hop_dist=2)
    print(data.x)
    pos = {
        i_n: g.nodes[big_from_small[i_n]][POS] for i_n in range(data.num_nodes)
    }

    f, ax = plt.subplots(1, 1)
    plot_graph_wo_pos_data(ax, data.edge_index, pos, data.x)
    plt.show()
