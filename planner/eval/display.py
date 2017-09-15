import numpy as np
from matplotlib import pyplot as plt

plt.style.use('bmh')


def plot_inputs(ax, agent_pos, idle_goals, jobs, grid):
    # Set grid lines to between the cells
    major_ticks = np.arange(0, len(grid[:, 0, 0]) + 1, 2)
    minor_ticks = np.arange(0, len(grid[:, 0, 0]) + 1, 1) + .5
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=0.2)
    # Make positive y pointing up
    ax.axis([-1, len(grid[:, 0]), -1, len(grid[:, 0])])
    # Show map
    plt.imshow(grid[:, :, 0] * -1, cmap="Greys", interpolation='nearest')
    # Agents
    agents = np.array(agent_pos)
    plt.scatter(agents[:, 0],
                agents[:, 1],
                s=np.full(agents.shape[0], 100),
                color='blue',
                alpha=.9)
    # Jobs
    for j in jobs:
        plt.arrow(x=j[0][0],
                  y=j[0][1],
                  dx=j[1][0] - j[0][0],
                  dy=j[1][1] - j[0][1],
                  head_width=.3, head_length=.7,
                  length_includes_head=True,
                  ec='r',
                  fill=False)
    # Fake for legend...
    plt.plot((0, 0), (.1, .1), 'r')

    # Idle Goals
    igs = []
    for ai in idle_goals:
        igs.append(ai[0])
    if igs:
        igs_array = np.array(igs)
        plt.scatter(igs_array[:, 0],
                    igs_array[:, 1],
                    s=np.full(igs_array.shape[0], 100),
                    color='g',
                    alpha=.9)
        # Legendary!
        plt.legend(["Transport Task", "Agent", "Idle Task"])
    else:
        plt.legend(["Transport Task", "Agent"])
    plt.title("State Variables")


def plot_results(ax, _agent_idle, _paths, agent_job, agent_pos, grid, idle_goals, jobs, title=''):
    from mpl_toolkits.mplot3d import Axes3D
    _ = Axes3D
    ax.axis([-1, len(grid[:, 0]), -1, len(grid[:, 0])])

    # Paths
    legend_str = []
    i = 0
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    assert _paths, "Paths have not been set"
    for _pathset in _paths:  # pathset per agent
        for p in _pathset:
            pa = np.array(p)
            ax.plot(xs=pa[:, 0],
                     ys=pa[:, 1],
                     zs=pa[:, 2],
                     color=colors[i])
            legend_str.append("Agent " + str(i))
        i += 1
    plt.legend(legend_str, loc=4)
    plt.title("Solution " + title)
    plt.tight_layout()