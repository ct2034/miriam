
import numpy as np
from matplotlib import pyplot as plt


def plot_env(ax, env):
    ax.imshow(np.swapaxes(env, 0, 1), cmap='Greys', origin='lower')


def plot_with_arrows(env, starts, goals):
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot()
    plot_env(ax, env)
    n_agents = len(starts)
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, n_agents+1)]
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
    plt.show()
