
from functools import reduce
from itertools import product

import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data

_ = Axes3D


def get_colors(n_agents):
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, n_agents+1)]
    return colors


def plot_env(ax, env):
    ax.imshow(np.swapaxes(env, 0, 1), cmap='Greys', origin='lower')


def plot_with_arrows(env, starts, goals):
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot()
    plot_env(ax, env)
    n_agents = len(starts)
    colors = get_colors(n_agents)
    for i_a in range(n_agents):
        ax.arrow(
            starts[i_a][0],
            starts[i_a][1],
            goals[i_a][0] - starts[i_a][0],
            goals[i_a][1] - starts[i_a][1],
            width=float(env.shape[0]/50),
            length_includes_head=True,
            linewidth=float(env.shape[0]/20),
            color=colors[i_a])


def plot_with_paths(env, paths):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis([-1, len(env[:, 0]), -1, len(env[:, 0])])
    ax.set_facecolor('white')
    ax.set_proj_type('ortho')

    assert env.shape[0] == env.shape[1], "assuming square map"
    width = env.shape[0] + 1

    # MAP
    xx, yy = np.meshgrid(
        np.arange(width),
        np.arange(width))
    zeros = np.zeros((width, width))

    facecolors = cm.Greys(np.array(np.swapaxes(env, 0, 1), dtype=float))
    # ax.contourf(xx, yy, np.swapaxes(env, 0, 1),
    #             antialiased=False, zdir='z', cmap=cm.Greys, alpha=0.8,
    #             zorder=0, levels=2, origin='lower')
    ax.plot_surface(xx, yy, zeros, rstride=1, cstride=1,
                    facecolors=facecolors, shade=False)

    ax.set_xlim(0, env.shape[0])
    ax.set_ylim(0, env.shape[1])

    # Paths
    colors = get_colors(len(paths))
    legend_str = []
    i = 0
    prop_cycle = plt.rcParams['axes.prop_cycle']
    assert paths is not None, "Paths have not been set"
    for p in paths:  # pathset per agent
        ax.plot(xs=p[:, 0] + .5,
                ys=p[:, 1] + .5,
                zs=p[:, 2],
                color=colors[i],
                alpha=1,
                zorder=100)
        legend_str.append("Agent " + str(i))
        i += 1

    # plt.legend(legend_str, bbox_to_anchor=(1, .95))
    plt.tight_layout()


def plot_schedule(data: dict):
    n_agents = len(data['indepAgentPaths'])
    colors = get_colors(n_agents)
    schedule = data['schedule']
    paths = []
    for i in range(n_agents):
        agent_str = 'agent'+str(i)
        assert agent_str in schedule.keys()
        path = np.array(list(
            map(lambda s: [s['x'], s['y'], s['t']],
                schedule[agent_str])
        ))
        paths.append(path)
    plot_with_paths(data['gridmap'], paths)


def plot_state(data: Data):
    """display state of GCN data with pos, edges and node features"""
    data_x = data.x
    data_edge_index = data.edge_index
    data_pos = data.pos

    n_nodes = data_x.shape[0]
    g = nx.Graph()
    g.add_edges_from(data_edge_index.T)

    label_dict = {}
    for i_n in range(n_nodes):
        data = list(map(float, data_x[i_n, :]))
        label_dict[i_n] = (f'[{data[0]:.2f}, {data[1]:.2f}, {data[2]:.2f}]\n' +
                           f'[{data[3]:.2f}, {data[4]:.2f}, {data[5]:.2f}]\n' +
                           f'[{data[6]:.2f}, {data[7]:.2f}, {data[8]:.2f}]')
    plt.figure()
    nx.draw(g, pos=data_pos.tolist())
    nx.draw_networkx_labels(g, pos=data_pos.tolist(),
                            labels=label_dict,
                            font_family='monospace', font_size=8)
