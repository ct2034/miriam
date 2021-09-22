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
    baserange = np.arange(environent.shape[0], step=2)  # type: ignore
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
        # spreading agents out across cells
        span = 0.8
        d_a = - span / 2 + i_a * span / (len(agents) - 1)

        # path
        line, = ax.plot(np.array(a.path)[:, 0] + d_a,
                        np.array(a.path)[:, 1] + d_a, color=colormap[i_a])
        line.set_label(str(i_a))
        # position
        if a.pos[0] != a.goal[0] or a.pos[1] != a.goal[1]:  # not at goal
            ax.plot(a.pos[0] + d_a, a.pos[1] + d_a, markersize=10,
                    marker='o', color=colormap[i_a])
        # goal
        ax.plot(a.goal[0] + d_a, a.goal[1] + d_a, markersize=10,
                marker='.', color=colormap[i_a])
        # blocked nodes
        blocks = np.array(list(a.blocked_nodes), dtype=int)
        if len(blocks) > 0:
            ax.plot(blocks[:, 0], blocks[:, 1], markersize=10,
                    marker='s', color=colormap[i_a], linewidth=0)

    ax.legend()
    plt.show()


def plot_evaluations(evaluations: Dict[PolicyType, np.ndarray],
                     evaluation_names: List[str], subtract_for_policy=None
                     ):  # pragma: no cover
    """Plot violinplots to compare values of different evaluations for 
    differnet policies.
    If `subtract_for_policy ` is defined, that policies result on all 
    evaluations will be substracted from all other policies values
    (for comparison)."""
    if subtract_for_policy is not None:
        policies = [p for p in evaluations.keys(
        ) if p != subtract_for_policy]
    else:
        policies = list(evaluations.keys())
    n_policies = len(policies)
    colormap = cm.tab10.colors
    data_shape = (n_policies, ) + list(evaluations.values())[0].shape
    data = np.empty(data_shape)
    i_success = evaluation_names.index('successful')

    policy_names = []
    subplot_basenr = 100 + 10 * len(evaluation_names) + 1
    all_successfull = [True] * data_shape[2]
    for i_p, policy in enumerate(policies):
        if subtract_for_policy is not None:
            data[i_p, :, :] = (evaluations[policy] -
                               evaluations[subtract_for_policy])
        else:
            data[i_p, :, :] = evaluations[policy]
        # overwrite success data:
        data[i_p, i_success, :] = evaluations[policy][i_success]
        all_successfull = np.logical_and(
            all_successfull,
            data[i_p, i_success, :])  # type: ignore
        policy_names.append(str(policy).replace('PolicyType.', ''))

    plt.figure(figsize=[16, 9])

    for i_e, evaluation_name in enumerate(evaluation_names):
        ax = plt.subplot(subplot_basenr + i_e)
        if i_e == i_success:
            run_choice = [True] * data_shape[2]
        else:
            run_choice = all_successfull
        try:
            parts = ax.violinplot(np.transpose(
                data[:, i_e, run_choice]), showmeans=True)
            for i_pc, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colormap[i_pc])
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                vp = parts[partname]
                vp.set_edgecolor('k')
        except ValueError:
            pass
        ax.set_title(evaluation_name)
        plt.xticks(range(1, n_policies + 1), policy_names)

    print(
        'success for all policies in '
        + f'{np.count_nonzero(all_successfull)} of {data_shape[2]} runs.')

    plt.show()
