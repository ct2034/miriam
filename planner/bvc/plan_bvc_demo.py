import logging
import random
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from multi_optim.multi_optim_run import RADIUS
from planner.bvc.plan_bvc import get_average_path_length, plan
from roadmaps.var_odrm_torch.var_odrm_torch import read_map, sample_points
from tools import ProgressBar

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("planner.bvc.plan_bvc").setLevel(logging.DEBUG)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    random.seed(0)
    rng = random.Random(0)
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/plain.png"
    map_img = read_map(map_fname)

    n_agents_s = range(2, 10, 1)
    n_trials = 10

    path_lengths = [list() for _ in range(len(n_agents_s))]
    success = [0, ] * len(n_agents_s)
    valid_paths_for_plotting = None

    pb = ProgressBar("eval", len(n_agents_s) * n_trials, 1)
    for i_na, n_agents in enumerate(n_agents_s):
        print(f"{n_agents=}")
        for i_t in range(n_trials):
            starts = sample_points(n_agents, map_img, rng).tolist()
            goals = sample_points(n_agents, map_img, rng).tolist()
            paths = plan(map_img, starts, goals, radius=RADIUS)
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
