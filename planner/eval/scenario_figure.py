from planner.cbs_ext.plan import plan, generate_config
from planner.milp.milp import plan_milp
from tools import load_map
from planner.eval.display import *
from mpl_toolkits.mplot3d import Axes3D
_ = Axes3D

config = generate_config()
config['finished_agents_block'] = True

if True:
    grid = load_map('mr_t.png')
    agent_pos = [(6, 3),
                 (3, 1)]
    jobs = [((5, 3), (5, 1), 0),
            ((4, 1), (4, 3), 0)]
    config['filename_pathsave'] = 'mr_t.pkl'
else:
    jobs = [((0, 0), (2, 6), 0),
            ((6, 2), (2, 0), 0),
            ((4, 6), (6, 2), 0),]
            #((7, 7), (4, 6), 0)]
    grid = load_map('ff.png')
    agent_pos = [(0, 0),
                 (2, 6),
                 (7, 7)]
    config['filename_pathsave'] = 'ff.pkl'

grid = np.repeat(grid[:, ::2, np.newaxis], 100, axis=2)

f = plt.figure()
ax1 = f.add_subplot(131)
f.set_size_inches(9, 3.5)
plot_inputs(ax1, agent_pos, [], jobs, grid)

tcbs_agent_job, tcbs_agent_idle, tcbs_paths = plan(
    agent_pos, jobs, [], [], grid, config, plot=False
)
ax2 = f.add_subplot(132, projection='3d')
plot_results(
    ax2, [], tcbs_paths, tcbs_agent_job, agent_pos, grid, [], jobs, 'TCBS'
)

minlp_agent_job, minlp_paths = plan_milp(agent_pos, jobs, grid, config)
ax3 = f.add_subplot(133, projection='3d')
plot_results(
    ax3, [], minlp_paths, minlp_agent_job, agent_pos, grid, [], jobs, 'MINLP'
)

f.savefig('scenario.png')