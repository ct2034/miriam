
from functools import reduce
from itertools import product

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
            starts[i_a][0] + .5,
            starts[i_a][1] + .5,
            goals[i_a][0] - starts[i_a][0],
            goals[i_a][1] - starts[i_a][1],
            width=1,
            length_includes_head=True,
            linewidth=0,
            color=colors[i_a])


def plot_with_paths(env, paths):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis([-1, len(env[:, 0]), -1, len(env[:, 0])])
    ax.set_facecolor('white')
    ax.set_proj_type('ortho')

    # MAP
    xx, yy = np.meshgrid(
        np.arange(env.shape[0]),
        np.arange(env.shape[1]))
    zeros = np.zeros(env.shape)

    facecolors = cm.Greys(np.array(np.swapaxes(env, 0, 1), dtype=np.float))
    # ax.contourf(xx, yy, np.swapaxes(env, 0, 1),
    #             antialiased=False, zdir='z', cmap=cm.Greys, alpha=0.8,
    #             zorder=0, levels=2, origin='lower')
    ax.plot_surface(xx, yy, zeros, rstride=1, cstride=1,
                    facecolors=facecolors, shade=False)

    # ax.set_xlim(0, env.shape[0])
    # ax.set_ylim(0, env.shape[1])

    # Paths
    colors = get_colors(len(paths))
    legend_str = []
    i = 0
    prop_cycle = plt.rcParams['axes.prop_cycle']
    assert paths, "Paths have not been set"
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
