from functools import reduce
from itertools import product

import matplotlib.animation as animation
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_ = Axes3D

plt.style.use("bmh")


def plot_inputs(ax, agent_pos, idle_goals, jobs, grid, title="Problem Configuration"):
    # Set grid lines to between the cells
    major_ticks = np.arange(0, len(grid[:, 0, 0]) + 1, 2)
    minor_ticks = np.arange(0, len(grid[:, 0, 0]) + 1, 1) + 0.5
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.2)
    # Make positive y pointing up
    ax.axis([-1, len(grid[:, 0]), -1, len(grid[:, 0])])
    # Show map
    ax.imshow(grid[:, :, 0] * -1, cmap="Greys", interpolation="nearest")
    # Agents
    agents = np.array(agent_pos)
    ax.scatter(
        agents[:, 0],
        agents[:, 1],
        s=np.full(agents.shape[0], 100),
        color="C0",
        alpha=0.9,
    )
    # Jobs
    for j in jobs:
        ax.arrow(
            x=j[0][0],
            y=j[0][1],
            dx=j[1][0] - j[0][0],
            dy=j[1][1] - j[0][1],
            head_width=0.25,
            head_length=0.4,
            width=0.05,
            length_includes_head=True,
            ec="C1",
            fc="C1",
            fill=True,
        )
    # Fake for legend...
    ax.plot((0, 0), (0.1, 0.1), "C1")

    # Idle Goals
    igs = []
    for ai in idle_goals:
        igs.append(ai[0])
    if igs:
        igs_array = np.array(igs)
        ax.scatter(
            igs_array[:, 0],
            igs_array[:, 1],
            s=np.full(igs_array.shape[0], 100),
            color="g",
            alpha=0.9,
        )
        # Legendary!
        ax.legend(["Transport Task", "Agent", "Idle Task"])
    else:
        ax.legend(["Transport Task", "Agent"])
    plt.title(title)


def plot_results(
    ax, _agent_idle, _paths, agent_job, agent_pos, grid, idle_goals, jobs, title=""
):
    from mpl_toolkits.mplot3d import Axes3D

    _ = Axes3D
    ax.axis([-1, len(grid[:, 0]), -1, len(grid[:, 0])])
    ax.set_facecolor("white")

    # Paths
    legend_str = []
    i = 0
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    assert _paths, "Paths have not been set"
    for _pathset in _paths:  # pathset per agent
        p = reduce(lambda a, b: a + b, _pathset, list())
        pa = np.array(p)
        ax.plot(xs=pa[:, 0], ys=pa[:, 1], zs=pa[:, 2], color=colors[i + 2])
        legend_str.append("Agent " + str(i))
        i += 1

    xx, yy = np.meshgrid(
        np.linspace(0.5, grid.shape[0] + 0.5, grid.shape[0] * 50),
        np.linspace(0.5, grid.shape[1] + 0.5, grid.shape[1] * 50),
    )

    img = np.zeros([xx.shape[0], yy.shape[1]])

    for x, y in product(range(len(xx[1, :])), range(len(yy[:, 1]))):
        try:
            img[x, y] = (
                grid[int(round(xx[0, x] - 1)), int(round(yy[y, 0] - 1)), 0] * -0.001
            )
        except IndexError:
            pass

    xx -= 1
    yy -= 1

    ax.contourf(xx, yy, img, antialiased=True, cmap=cm.Greys, alpha=0.8)

    plt.legend(legend_str, bbox_to_anchor=(1, 0.95))
    plt.title("Solution " + title)
    plt.tight_layout()


def update_lines(num, data, lines):
    print("Num: " + str(num))
    for i in range(data.shape[0]):
        lines[i].set_data(data[i, ..., :num])
        lines[i].set_markevery([num - 1])
    return lines


def animate_results(
    fig, _agent_idle, _paths, agent_job, agent_pos, grid, idle_goals, jobs, title=""
):
    ax = fig.add_subplot(111)

    major_ticks = np.arange(0, len(grid[:, 0, 0]) + 1, 2)
    minor_ticks = np.arange(0, len(grid[:, 0, 0]) + 1, 1) + 0.5
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.2)
    ax.axis([-1, len(grid[:, 0]), -1, len(grid[:, 0])])

    # Prepare Data
    n_a = len(agent_pos)
    pa = []
    for _pathset in _paths:  # pathset per agent
        p = reduce(lambda a, b: a + b, _pathset, list())
        if len(p) == 15:
            p.pop()
        pa.append(np.array(p))
    data = np.zeros([n_a, pa[0].shape[0], 2])
    for i in range(n_a):
        data[i, :, :] = pa[i][:, :2]
    data = np.swapaxes(data, 1, 2)

    # Plot
    ax.imshow(grid[:, :, 0] * -1, cmap="Greys", interpolation="nearest")

    ls = [None] * n_a
    for i in range(n_a):
        (ls[i],) = ax.plot([], [], "C1:o")
        ls[i].set_markerfacecolor("C0")
        ls[i].set_markeredgecolor("C0")
        ls[i].set_markersize(10)
    plt.title("Solution " + title)
    plt.tight_layout()

    line_ani = animation.FuncAnimation(
        fig,
        update_lines,
        frames=data.shape[2] + 1,
        fargs=(data, ls),
        interval=1000,
        blit=True,
    )
    return line_ani
