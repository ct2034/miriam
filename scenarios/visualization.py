
from functools import reduce
from itertools import product
from typing import List
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data
from definitions import POS
from scenarios.types import POTENTIAL_ENV_TYPE, is_gridmap, is_roadmap

_ = Axes3D


def get_colors(n_agents) -> List[float]:
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, n_agents+1)]
    return colors


def plot_env(ax, env: POTENTIAL_ENV_TYPE):
    if is_gridmap(env):
        ax.imshow(np.swapaxes(env, 0, 1), cmap='Greys', origin='lower')
        return None
    elif is_roadmap(env):
        pos_s = nx.get_node_attributes(env, POS)
        nx.draw_networkx(env,
                         pos=pos_s, ax=ax,
                         node_color='k', edge_color='grey',
                         node_size=10, with_labels=False)
        return pos_s


def plot_env_with_arrows(env: POTENTIAL_ENV_TYPE, starts, goals):
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot()
    pos_s = plot_env(ax, env)
    if is_gridmap(env):
        assert pos_s is None
        start_pos_s = starts
        goal_pos_s = goals
    elif is_roadmap(env):
        assert pos_s is not None
        start_pos_s = list(map(lambda s: pos_s[s], starts))
        goal_pos_s = list(map(lambda g: pos_s[g], goals))
    n_agents = len(start_pos_s)
    colors = get_colors(n_agents)
    for i_a in range(n_agents):
        ax.arrow(
            start_pos_s[i_a][0],
            start_pos_s[i_a][1],
            goal_pos_s[i_a][0] - start_pos_s[i_a][0],
            goal_pos_s[i_a][1] - start_pos_s[i_a][1],
            width=.004,
            length_includes_head=True,
            linewidth=.002,
            color=colors[i_a])
    ax.set_aspect('equal')


def plot_with_paths(env, paths, fig=plt.figure()):
    ax = fig.add_subplot(111, projection='3d')
    colors = get_colors(len(paths))
    legend_str = []
    if is_gridmap(env):
        if len(paths[0][0]) == 2:  # not timed
            paths_timed = []
            for ap in paths:
                ap_t = []
                for t, p in enumerate(ap):
                    ap_t.append(p + (t,))
                paths_timed.append(ap_t)
            paths = np.array(paths_timed)
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
        t = 0
        prop_cycle = plt.rcParams['axes.prop_cycle']
        assert paths is not None, "Paths have not been set"
        for p in paths:  # pathset per agent
            ax.plot(xs=p[:, 0] + .5,
                    ys=p[:, 1] + .5,
                    zs=p[:, 2],
                    color=colors[t],
                    alpha=1,
                    zorder=100)
            legend_str.append("Agent " + str(t))
            t += 1
    elif is_roadmap(env):
        if isinstance(paths[0][0], int):  # not timed
            paths_timed = []
            for ap in paths:
                ap_t = []
                for t, p in enumerate(ap):
                    ap_t.append((p, t))
                paths_timed.append(ap_t)
            paths = paths_timed
        pos = nx.get_node_attributes(env, POS)
        for e in env.edges():
            ax.plot([pos[e[0]][0], pos[e[1]][0]],
                    [pos[e[0]][1], pos[e[1]][1]],
                    [0, 0],
                    color='grey',
                    alpha=0.5,
                    linewidth=0.5,
                    zorder=10)
        assert paths is not None, "Paths have not been set"
        for t, path in enumerate(paths):
            # pathset per agent
            p = np.array(list(map(
                lambda s: tuple(pos[s[0]]) + (s[1],), path)))
            ax.plot(xs=p[:, 0],
                    ys=p[:, 1],
                    zs=p[:, 2],
                    color=colors[t],
                    alpha=1,
                    zorder=100,
                    marker="o")
            legend_str.append("Agent " + str(t))
            # projection lines to graph
            for pose in p.tolist():
                ax.plot(
                    [pose[0], pose[0]],
                    [pose[1], pose[1]],
                    [pose[2], 0.],
                    color=colors[t],
                    alpha=0.5,
                    linewidth=0.5,
                    zorder=50
                )
        ax.grid(False)
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
