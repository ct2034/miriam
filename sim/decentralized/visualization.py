from typing import Dict, List

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from .policy import PolicyType


def plot_env_agents(environent: np.ndarray,
                    agents: np.ndarray):  # pragma: no cover
    """Plot the environment map with `x` coordinates to the right, `y` up.
    Occupied by colorful agents and their paths."""
    # map
    image = environent * -.5 + 1
    image = np.swapaxes(image, 0, 1)
    fig, ax = plt.subplots()
    c = ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.set_aspect('equal')
    baserange = np.arange(environent.shape[0], step=2)
    ax.set_xticks(baserange)
    ax.set_xticklabels(map(str, baserange))
    ax.set_yticks(baserange)
    ax.set_yticklabels(map(str, baserange))
    colormap = [cm.hsv(i/len(agents)) for i in range(len(agents))]

    # gridlines
    for i_x in range(environent.shape[0]-1):
        ax.plot(
            [i_x + .5] * 2,
            [-.5, environent.shape[1]-.5],
            color='dimgrey',
            linewidth=.5
        )
    for i_y in range(environent.shape[1]-1):
        ax.plot(
            [-.5, environent.shape[0]-.5],
            [i_y + .5] * 2,
            color='dimgrey',
            linewidth=.5
        )

    # agents
    for i_a, a in enumerate(agents):
        # path
        ax.plot(a.path[:, 0], a.path[:, 1], color=colormap[i_a])
        # position
        if a.pos[0] != a.goal[0] or a.pos[1] != a.goal[1]:  # not at goal
            ax.plot(a.pos[0], a.pos[1], markersize=10,
                    marker='o', color=colormap[i_a])
        # goal
        ax.plot(a.goal[0], a.goal[1], markersize=10,
                marker='.', color=colormap[i_a])
        # blocked nodes
        blocks = np.array(list(a.filter_blocked_nodes), dtype=int)
        if len(blocks) > 0:
            ax.plot(blocks[:, 0], blocks[:, 1], markersize=10,
                    marker='s', color=colormap[i_a], linewidth=0)

    plt.show()


def plot_evaluations(evaluations: Dict[PolicyType, np.ndarray],
                     evaluation_names: List[str]):  # pragma: no cover
    """Plot violinplots to compare values of different evaluations for 
    differnet policies."""
    n_policies = len(evaluations.keys())
    colormap = cm.tab10.colors
    data_shape = (n_policies, ) + list(evaluations.values())[0].shape
    data = np.empty(data_shape)
    i_success = evaluation_names.index('successful')

    policy_names = []
    subplot_basenr = 100 + 10 * len(evaluation_names) + 1
    all_successfull = [True] * data_shape[2]
    for i_p, policy in enumerate(evaluations.keys()):
        data[i_p, :, :] = evaluations[policy]
        all_successfull = np.logical_and(
            all_successfull,
            data[i_p, i_success, :])
        policy_names.append(str(policy).replace('PolicyType.', ''))

    plt.figure(figsize=[16, 9])

    for i_e, evaluation_name in enumerate(evaluation_names):
        ax = plt.subplot(subplot_basenr + i_e)
        if i_e == i_success:
            run_choice = [True] * data_shape[2]
        else:
            run_choice = all_successfull
        parts = ax.violinplot(np.transpose(
            data[:, i_e, run_choice]), showmeans=True)
        for i_pc, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colormap[i_pc])
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vp = parts[partname]
            vp.set_edgecolor('k')
        ax.set_title(evaluation_name)
        plt.xticks(range(1, n_policies + 1), policy_names)

    print(
        'success for all policies in '
        + f'{np.count_nonzero(all_successfull)} of {data_shape[2]} runs.')

    plt.show()
