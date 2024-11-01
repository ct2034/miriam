import numpy as np

from planner.eval.display import plot_inputs, plot_results
from planner.greedy.greedy import plan_greedy
from tools import load_map

if __name__ == "__main__":
    _map = load_map("../map2.png")
    _map = _map[:, ::2]
    grid = np.repeat(_map[:, :, np.newaxis], 100, axis=2)

    # input
    agent_pos = [(0, 0), (1, 0), (2, 0)]
    jobs = [
        ((0, 8), (0, 2), 0),
        ((1, 8), (2, 4), 0),
        ((2, 8), (4, 8), 0),
        ((7, 6), (3, 8), 0),
        ((8, 7), (8, 2), 0),
        ((3, 4), (5, 8), 0),
    ]
    idle_goals = []

    fig = plot_inputs(agent_pos, idle_goals, jobs, grid, show=False)

    (res_agent_job, res_paths) = plan_greedy(agent_pos, jobs, grid, "greedy.pkl")

    plot_results([], res_paths, res_agent_job, agent_pos, fig, grid, idle_goals, jobs)
