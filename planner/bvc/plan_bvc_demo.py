import random
from tracemalloc import start
from turtle import width
from typing import List

import numpy as np
from definitions import INVALID, MAP_IMG
from matplotlib import pyplot as plt
from planner.bvc.plan_bvc import get_average_path_length, plan
from roadmaps.var_odrm_torch.var_odrm_torch import read_map
from tools import ProgressBar

if __name__ == "__main__":
    random.seed(0)
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/plain.png"
    map_img_np: np.ndarray = np.array(read_map(map_fname))
    width = map_img_np.shape[1]
    print(f"{width=}")
    desired_width = width // 32
    print(f"{desired_width=}")
    bin_size = width // desired_width
    map_img_np = map_img_np.reshape((desired_width, bin_size,
                                     desired_width, bin_size)).max(3).max(1)
    map_img = tuple([tuple(map_img_np[i, :].tolist())
                     for i in range(map_img_np.shape[0])])

    possible_starts_and_goals = [
        [.7, .6], [.7, .4],
        [.6, .7], [.4, .7],
        [.3, .6], [.3, .4],
        [.6, .3], [.4, .3],
    ]
    for i_p in range(len(possible_starts_and_goals)):
        possible_starts_and_goals[i_p][0] += random.gauss(0, .1)
        possible_starts_and_goals[i_p][1] += random.gauss(0, .1)
    n_agents_s = range(1, len(possible_starts_and_goals))
    n_trials = 10

    path_lengths = [list() for _ in range(len(n_agents_s))]
    success = [0, ] * len(n_agents_s)
    valid_paths_for_plotting = None

    pb = ProgressBar("eval", len(n_agents_s) * n_trials, 1)
    for i_na, n_agents in enumerate(n_agents_s):
        for i_t in range(n_trials):
            starts = random.sample(possible_starts_and_goals, k=n_agents)
            goals: List = []
            for i_a in range(n_agents):
                remaining_possible_goals = possible_starts_and_goals.copy()
                remaining_possible_goals.remove(starts[i_a])
                for g in goals:
                    if g in remaining_possible_goals:
                        remaining_possible_goals.remove(g)
                goals.append(random.choice(remaining_possible_goals))
            paths = plan(map_img, starts, goals, radius=0.01)
            if isinstance(paths, np.ndarray):
                success[i_na] += 1
                path_lengths[i_na].append(get_average_path_length(paths))
                valid_paths_for_plotting = paths
            pb.progress()
        if len(path_lengths[i_na]) == 0:
            path_lengths[i_na] = [0.0, ]
    pb.end()

    # plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), sharex=True)
    ax1.bar(n_agents_s, success)
    ax1.set_title("Success rate")

    ax2.violinplot(path_lengths, n_agents_s, showmeans=True)
    ax2.set_title("Average path length")
    ax2.set_xlabel("Number of agents")

    plt.savefig("planner/bvc/bvc_eval.png")

    # figure for paths
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # show map
    try:
        xx, yy = np.meshgrid(np.linspace(0, 1, len(map_img)),
                             np.linspace(0, 1, len(map_img)))
        z = np.zeros_like(xx)
        img_data = np.array(map_img) / -255. + 1
        ax.contourf(xx, yy, img_data, len(map_img),
                    zdir='z', offset=0.5, cmap=plt.cm.Greys, zorder=200)
    except Exception as e:
        pass

    # show paths
    for i_a in range(len(valid_paths_for_plotting)):
        ax.plot(
            valid_paths_for_plotting[i_a, :, 0],
            valid_paths_for_plotting[i_a, :, 1],
            range(valid_paths_for_plotting.shape[1]),
            label=f"robot_{i_a}",
            zorder=100+i_a)
    ax.set_zlim((0., valid_paths_for_plotting.shape[1]))
    ax.legend()

    plt.savefig("planner/bvc/bvc_eval_paths.png")
