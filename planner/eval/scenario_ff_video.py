import matplotlib.pyplot as plt
import numpy as np

from planner.eval.display import animate_results, plot_inputs
from planner.milp.milp import plan_milp
from planner.tcbs.plan import generate_config, plan
from tools import load_map

if __name__ == "__main__":
    config = generate_config()
    jobs = [
        ((0, 0), (2, 6), 0),
        ((6, 2), (2, 0), 0),
        ((4, 6), (6, 2), 0),
    ]
    grid = load_map("ff.png")
    agent_pos = [(0, 0), (2, 6), (7, 7)]
    config["filename_pathsave"] = "ff.pkl"

    grid = np.repeat(grid[:, ::2, np.newaxis], 100, axis=2)

    params = {"legend.fontsize": "small"}
    plt.rcParams.update(params)

    fc = plt.figure()
    ax1 = fc.add_subplot(131)
    fc.set_size_inches(9, 3.5)
    plot_inputs(ax1, agent_pos, [], jobs, grid)
    fc.savefig("scenario_ff_config.png")

    tcbs_agent_job, tcbs_agent_idle, tcbs_paths = plan(
        agent_pos, jobs, [], [], grid, config, plot=False
    )

    minlp_agent_job, minlp_paths = plan_milp(agent_pos, jobs, grid, config)

    print((tcbs_paths, tcbs_agent_job, minlp_paths, minlp_agent_job, agent_pos, jobs))

    f = plt.figure()
    f.set_size_inches(8, 4.5)
    ani = animate_results(
        f, [], tcbs_paths, tcbs_agent_job, agent_pos, grid, [], jobs, "TCBS"
    )
    ani.save("scenario_ff_tcbs.mp4", writer="ffmpeg", fps=10)

    f = plt.figure()
    f.set_size_inches(8, 4.5)
    ani = animate_results(
        f, [], minlp_paths, minlp_agent_job, agent_pos, grid, [], jobs, "MINLP"
    )
    ani.save("scenario_ff_minlp.mp4", writer="ffmpeg", fps=10)
    # plt.show()
