import hashlib

from planner.tcbs_test import get_data_random
from planner.eval.display import plot_inputs, plot_results
from planner.tcbs.plan import plan, generate_config
from planner.milp.milp import plan_milp

import matplotlib.pyplot as plt

from tools import get_map_str

params = get_data_random(330,
                         map_res=8,
                         map_fill_perc=30,
                         agent_n=4,
                         job_n=4,
                         idle_goals_n=0)
agent_pos, grid, idle_goals, jobs = params

mapstr = get_map_str(grid)
maphash = str(hashlib.md5(mapstr.encode('utf-8')).hexdigest())[:8]
fname = "planner/eval/cache/" + str(maphash) + '.pkl'  # unique filename based on map

config = generate_config()
config['finished_agents_block'] = True
config['filename_pathsave'] = fname

tcbs_agent_job, tcbs_agent_idle, tcbs_paths = plan(agent_pos, jobs, [], [], grid, config, plot=False)
f = plt.figure()
ax0= f.add_subplot(111)
plot_results(ax0, tcbs_agent_idle, tcbs_paths, tcbs_agent_job, agent_pos, grid, [], jobs, "tcbs")

milp_agent_job, milp_agent_idle, milp_paths = plan_milp(agent_pos, jobs, [], [], grid, config, plot=False)
f1 = plt.figure()
ax1= f1.add_subplot(111)
plot_results(ax1, milp_agent_idle, milp_paths, milp_agent_job, agent_pos, grid, [], jobs, "milp")

print(str((tcbs_paths, tcbs_agent_job, milp_paths, milp_agent_job, agent_pos, jobs)))

params = {'legend.fontsize': 'small'}
plt.rcParams.update(params)

f = plt.figure()
ax1 = f.add_subplot(111)
f.set_size_inches(4, 4)
plot_inputs(ax1, agent_pos, [], jobs, grid, title='')

f.savefig('random.png')
f.set_size_inches(19.20, 10.80)
plt.title("Problem Configuration")
plt.tight_layout()
f.savefig('video/random_problem.png')
